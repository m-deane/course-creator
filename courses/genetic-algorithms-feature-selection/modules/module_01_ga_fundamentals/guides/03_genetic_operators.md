# Genetic Operators: Crossover and Mutation

## In Brief

Crossover combines genetic material from two parents to create offspring, enabling the exchange of beneficial feature combinations. Mutation introduces random changes to maintain diversity and explore new regions of the search space. Together, these operators balance exploitation (crossover) with exploration (mutation).

## Key Insight

The effectiveness of genetic operators depends on the problem structure. For feature selection, uniform crossover often outperforms single-point crossover because feature interactions are typically non-positional. Mutation rate should be inversely proportional to chromosome length to maintain approximately one change per individual.

## Formal Definition

### Crossover

Given parents $\mathbf{p}_1, \mathbf{p}_2 \in \{0,1\}^n$, crossover produces offspring $\mathbf{o}_1, \mathbf{o}_2$ by recombining genetic material:

**Single-Point Crossover** at position $k$:
$$\mathbf{o}_1 = [\mathbf{p}_1[1:k], \mathbf{p}_2[k+1:n]]$$
$$\mathbf{o}_2 = [\mathbf{p}_2[1:k], \mathbf{p}_1[k+1:n]]$$

**Uniform Crossover** with swap probability $p_{swap}$:
$$o_{1,i} = \begin{cases} p_{2,i} & \text{with probability } p_{swap} \\ p_{1,i} & \text{otherwise} \end{cases}$$

**Applied with probability** $p_c \in [0.6, 0.95]$ (typically 0.8)

### Mutation

Given individual $\mathbf{x} \in \{0,1\}^n$, mutation produces $\mathbf{x}'$:

**Bit-Flip Mutation** at rate $p_m$:
$$x'_i = \begin{cases} 1 - x_i & \text{with probability } p_m \\ x_i & \text{otherwise} \end{cases}$$

**Typical mutation rate**: $p_m = 1/n$ (one bit flip per individual on average)

### Building Block Hypothesis

Crossover works by combining **building blocks** (schemata): short, low-order, high-fitness patterns.

A schema $H$ is a template: $H = [1, *, 0, *, *]$ matches chromosomes $[1,0,0,1,0]$, $[1,1,0,0,1]$, etc.

Crossover preserves building blocks when crossover point falls outside the schema.

## Intuitive Explanation

**Crossover** is like combining recipes from two chefs. If Chef A makes great appetizers and Chef B makes great desserts, combining their recipes might yield a meal that's great overall. In feature selection, one parent might have good features for capturing linear trends, another for capturing seasonality—their child might capture both.

**Mutation** is like random experimentation. Occasionally change an ingredient to try something new. Without mutation, you can only recombine existing ingredients—you might miss discovering that cilantro (a feature not in any parent) makes the dish perfect.

**Crossover vs. Mutation trade-off:**
- High crossover, low mutation: Fast convergence, may get stuck (all recipes become similar)
- Low crossover, high mutation: Slow convergence, maintains diversity (random search)
- Balanced (typical): Crossover exploits known good solutions, mutation explores new ones

## Code Implementation

### Crossover Operators

```python
import numpy as np
from typing import Tuple, Protocol, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class Individual(Protocol):
    """Protocol for binary-encoded individual."""
    chromosome: np.ndarray
    fitness: Optional[float]
    def copy(self) -> 'Individual': ...


@dataclass
class BinaryIndividual:
    """Concrete implementation for examples."""
    chromosome: np.ndarray
    fitness: Optional[float] = None

    def copy(self) -> 'BinaryIndividual':
        return BinaryIndividual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness
        )


class CrossoverType(Enum):
    """Supported crossover types."""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    SCATTERED = "scattered"


def single_point_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_prob: float = 0.8
) -> Tuple[Individual, Individual]:
    """
    Single-point crossover.

    Parameters
    ----------
    parent1, parent2 : Individual
        Parent individuals
    crossover_prob : float
        Probability of performing crossover (vs. returning clones)

    Returns
    -------
    Tuple[Individual, Individual]
        Two offspring

    Examples
    --------
    >>> p1 = BinaryIndividual(np.array([1,1,1,1,1]))
    >>> p2 = BinaryIndividual(np.array([0,0,0,0,0]))
    >>> # Crossover at position 2: [1,1|1,1,1] × [0,0|0,0,0]
    >>> # Offspring: [1,1,0,0,0] and [0,0,1,1,1]
    """
    # Apply crossover with probability
    if np.random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)

    # Choose random crossover point (not at extremes)
    point = np.random.randint(1, n)

    # Create offspring
    child1_chrom = np.concatenate([
        parent1.chromosome[:point],
        parent2.chromosome[point:]
    ])
    child2_chrom = np.concatenate([
        parent2.chromosome[:point],
        parent1.chromosome[point:]
    ])

    return (
        BinaryIndividual(chromosome=child1_chrom),
        BinaryIndividual(chromosome=child2_chrom)
    )


def two_point_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_prob: float = 0.8
) -> Tuple[Individual, Individual]:
    """
    Two-point crossover.

    Swaps the segment between two random points.

    Examples
    --------
    >>> # [1,1|1,1|1] × [0,0|0,0|0]
    >>> # Swap middle segment
    >>> # Offspring: [1,1,0,0,1] and [0,0,1,1,0]
    """
    if np.random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)

    # Choose two distinct crossover points
    points = sorted(np.random.choice(n, size=2, replace=False))
    point1, point2 = points[0], points[1]

    # Create offspring by swapping middle segment
    child1_chrom = parent1.chromosome.copy()
    child1_chrom[point1:point2] = parent2.chromosome[point1:point2]

    child2_chrom = parent2.chromosome.copy()
    child2_chrom[point1:point2] = parent1.chromosome[point1:point2]

    return (
        BinaryIndividual(chromosome=child1_chrom),
        BinaryIndividual(chromosome=child2_chrom)
    )


def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_prob: float = 0.8,
    swap_prob: float = 0.5
) -> Tuple[Individual, Individual]:
    """
    Uniform crossover (best for feature selection).

    Each gene independently inherited from random parent.

    Parameters
    ----------
    swap_prob : float
        Probability of swapping each gene
        0.5 = equal inheritance from both parents

    Notes
    -----
    Uniform crossover is most effective for feature selection because:
    1. No positional bias (features can be reordered)
    2. Can combine any subset of features from parents
    3. Maximum mixing of genetic material

    Examples
    --------
    >>> # Each position randomly inherits from p1 or p2
    >>> # [1,1,1,1,1] × [0,0,0,0,0]
    >>> # Possible offspring: [1,0,1,0,1], [0,1,1,0,0], etc.
    """
    if np.random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)

    # Create random swap mask
    mask = np.random.random(n) < swap_prob

    # Apply mask
    child1_chrom = np.where(mask, parent2.chromosome, parent1.chromosome)
    child2_chrom = np.where(mask, parent1.chromosome, parent2.chromosome)

    return (
        BinaryIndividual(chromosome=child1_chrom),
        BinaryIndividual(chromosome=child2_chrom)
    )


def scattered_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_prob: float = 0.8,
    n_points: Optional[int] = None
) -> Tuple[Individual, Individual]:
    """
    Scattered (multi-point) crossover.

    Generalizes two-point to k-point crossover.

    Parameters
    ----------
    n_points : int, optional
        Number of crossover points
        If None, chooses random number between 2 and n/2
    """
    if np.random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)

    # Choose number of points
    if n_points is None:
        n_points = np.random.randint(2, max(3, n // 2))

    # Choose random crossover points
    points = sorted(np.random.choice(n, size=min(n_points, n-1), replace=False))

    # Alternate between parents at each point
    child1_chrom = parent1.chromosome.copy()
    child2_chrom = parent2.chromosome.copy()

    swap = False
    prev_point = 0

    for point in points:
        if swap:
            child1_chrom[prev_point:point] = parent2.chromosome[prev_point:point]
            child2_chrom[prev_point:point] = parent1.chromosome[prev_point:point]
        swap = not swap
        prev_point = point

    # Handle last segment
    if swap:
        child1_chrom[prev_point:] = parent2.chromosome[prev_point:]
        child2_chrom[prev_point:] = parent1.chromosome[prev_point:]

    return (
        BinaryIndividual(chromosome=child1_chrom),
        BinaryIndividual(chromosome=child2_chrom)
    )
```

### Mutation Operators

```python
def bit_flip_mutation(
    individual: Individual,
    mutation_rate: Optional[float] = None,
    min_features: int = 1
) -> Individual:
    """
    Standard bit-flip mutation.

    Parameters
    ----------
    individual : Individual
        Individual to mutate
    mutation_rate : float, optional
        Probability of flipping each bit
        If None, uses 1/n (one flip per individual on average)
    min_features : int
        Minimum number of features that must be selected

    Returns
    -------
    Individual
        Mutated individual (new instance)

    Notes
    -----
    Rule of thumb: mutation_rate = 1/n
    This ensures approximately 1 bit flip per individual
    """
    mutant = individual.copy()
    n = len(mutant.chromosome)

    if mutation_rate is None:
        mutation_rate = 1.0 / n

    # Flip each bit with probability mutation_rate
    for i in range(n):
        if np.random.random() < mutation_rate:
            mutant.chromosome[i] = 1 - mutant.chromosome[i]

    # Enforce minimum features constraint
    while np.sum(mutant.chromosome) < min_features:
        # Turn on a random bit
        zero_indices = np.where(mutant.chromosome == 0)[0]
        if len(zero_indices) > 0:
            mutant.chromosome[np.random.choice(zero_indices)] = 1
        else:
            break

    mutant.fitness = None  # Invalidate fitness
    return mutant


def swap_mutation(
    individual: Individual,
    n_swaps: int = 1
) -> Individual:
    """
    Swap mutation - preserves number of selected features.

    Swaps n_swaps pairs of (selected, unselected) features.
    Useful when target feature count is important.

    Parameters
    ----------
    n_swaps : int
        Number of feature pairs to swap

    Examples
    --------
    >>> ind = BinaryIndividual(np.array([1,1,0,0,1,0]))  # 3 features
    >>> mutant = swap_mutation(ind, n_swaps=1)
    >>> np.sum(mutant.chromosome)  # Still 3 features
    3
    """
    mutant = individual.copy()

    for _ in range(n_swaps):
        selected = np.where(mutant.chromosome == 1)[0]
        unselected = np.where(mutant.chromosome == 0)[0]

        if len(selected) > 0 and len(unselected) > 0:
            # Choose one selected and one unselected feature
            turn_off = np.random.choice(selected)
            turn_on = np.random.choice(unselected)

            # Swap them
            mutant.chromosome[turn_off] = 0
            mutant.chromosome[turn_on] = 1

    mutant.fitness = None
    return mutant


def adaptive_mutation(
    individual: Individual,
    generation: int,
    max_generations: int,
    min_rate: float = 0.001,
    max_rate: float = 0.1,
    schedule: str = 'linear'
) -> Individual:
    """
    Adaptive mutation with decreasing rate.

    Parameters
    ----------
    generation : int
        Current generation number
    max_generations : int
        Total number of generations
    min_rate, max_rate : float
        Mutation rate bounds
    schedule : str
        'linear', 'exponential', or 'cosine'

    Notes
    -----
    High mutation early: broad exploration
    Low mutation late: fine-tuning
    """
    progress = generation / max_generations

    if schedule == 'linear':
        rate = max_rate - progress * (max_rate - min_rate)
    elif schedule == 'exponential':
        rate = max_rate * np.exp(-5 * progress)
        rate = max(rate, min_rate)
    elif schedule == 'cosine':
        rate = min_rate + 0.5 * (max_rate - min_rate) * (1 + np.cos(np.pi * progress))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return bit_flip_mutation(individual, mutation_rate=rate)


def inversion_mutation(
    individual: Individual,
    mutation_prob: float = 0.1
) -> Individual:
    """
    Inversion mutation - reverses a random segment.

    Less common for feature selection, more useful for
    permutation-based problems.

    Examples
    --------
    >>> # [1,1,0,0,1,0]
    >>> # Invert positions 1-4: [1,0,0,1,1,0]
    """
    if np.random.random() > mutation_prob:
        return individual.copy()

    mutant = individual.copy()
    n = len(mutant.chromosome)

    if n > 1:
        # Choose two points
        points = sorted(np.random.choice(n, size=2, replace=False))
        start, end = points[0], points[1]

        # Reverse segment
        mutant.chromosome[start:end+1] = mutant.chromosome[start:end+1][::-1]

    mutant.fitness = None
    return mutant
```

### Combined Operators and Analysis

```python
def apply_genetic_operators(
    parent1: Individual,
    parent2: Individual,
    crossover_type: CrossoverType = CrossoverType.UNIFORM,
    crossover_prob: float = 0.8,
    mutation_rate: Optional[float] = None,
    min_features: int = 1
) -> Tuple[Individual, Individual]:
    """
    Apply crossover followed by mutation (standard GA).

    Parameters
    ----------
    parent1, parent2 : Individual
        Parent individuals
    crossover_type : CrossoverType
        Type of crossover to apply
    crossover_prob : float
        Probability of crossover
    mutation_rate : float, optional
        Mutation rate (default: 1/n)
    min_features : int
        Minimum features to select

    Returns
    -------
    Tuple[Individual, Individual]
        Two offspring after crossover and mutation
    """
    # Crossover
    if crossover_type == CrossoverType.SINGLE_POINT:
        child1, child2 = single_point_crossover(parent1, parent2, crossover_prob)
    elif crossover_type == CrossoverType.TWO_POINT:
        child1, child2 = two_point_crossover(parent1, parent2, crossover_prob)
    elif crossover_type == CrossoverType.UNIFORM:
        child1, child2 = uniform_crossover(parent1, parent2, crossover_prob)
    elif crossover_type == CrossoverType.SCATTERED:
        child1, child2 = scattered_crossover(parent1, parent2, crossover_prob)
    else:
        raise ValueError(f"Unknown crossover type: {crossover_type}")

    # Mutation
    child1 = bit_flip_mutation(child1, mutation_rate, min_features)
    child2 = bit_flip_mutation(child2, mutation_rate, min_features)

    return child1, child2


def analyze_operator_effects():
    """
    Analyze effects of different operators on offspring diversity.
    """
    import matplotlib.pyplot as plt

    n_features = 20
    n_trials = 1000

    # Create two distinct parents
    parent1 = BinaryIndividual(np.zeros(n_features, dtype=int))
    parent1.chromosome[:10] = 1  # First half selected

    parent2 = BinaryIndividual(np.zeros(n_features, dtype=int))
    parent2.chromosome[10:] = 1  # Second half selected

    print("Parent Analysis")
    print("=" * 60)
    print(f"Parent 1: {parent1.chromosome}")
    print(f"Parent 2: {parent2.chromosome}")
    print(f"Hamming distance: {np.sum(parent1.chromosome != parent2.chromosome)}")

    # Test different crossover operators
    operators = {
        'Single-Point': lambda: single_point_crossover(parent1, parent2, 1.0),
        'Two-Point': lambda: two_point_crossover(parent1, parent2, 1.0),
        'Uniform': lambda: uniform_crossover(parent1, parent2, 1.0),
        'Scattered': lambda: scattered_crossover(parent1, parent2, 1.0),
    }

    results = {}

    for name, op_func in operators.items():
        offspring_patterns = []
        num_selected = []

        for _ in range(n_trials):
            child1, child2 = op_func()
            offspring_patterns.append(child1.chromosome.tobytes())
            num_selected.append(child1.chromosome.sum())

        unique_offspring = len(set(offspring_patterns))
        avg_selected = np.mean(num_selected)
        std_selected = np.std(num_selected)

        results[name] = {
            'unique': unique_offspring,
            'avg_selected': avg_selected,
            'std_selected': std_selected
        }

    # Print results
    print(f"\n{'Operator':<15} {'Unique':<10} {'Avg Selected':<15} {'Std Selected':<15}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<15} {stats['unique']:<10} "
              f"{stats['avg_selected']:<15.2f} {stats['std_selected']:<15.2f}")


def test_mutation_rates():
    """
    Test effect of mutation rate on diversity and fitness preservation.
    """
    n_features = 50
    n_trials = 1000

    # Create individual
    original = BinaryIndividual(np.zeros(n_features, dtype=int))
    original.chromosome[np.random.choice(n_features, size=10, replace=False)] = 1

    mutation_rates = [0.001, 0.01, 0.02, 0.05, 0.1, 0.2]

    print("\nMutation Rate Analysis")
    print("=" * 70)
    print(f"{'Rate':<10} {'Avg Changes':<15} {'Std Changes':<15} {'Avg Selected':<15}")
    print("-" * 70)

    for rate in mutation_rates:
        changes = []
        num_selected = []

        for _ in range(n_trials):
            mutant = bit_flip_mutation(original, mutation_rate=rate)
            n_changes = np.sum(original.chromosome != mutant.chromosome)
            changes.append(n_changes)
            num_selected.append(mutant.chromosome.sum())

        print(f"{rate:<10.3f} {np.mean(changes):<15.2f} "
              f"{np.std(changes):<15.2f} {np.mean(num_selected):<15.2f}")


def visualize_building_blocks():
    """
    Visualize how crossover preserves building blocks.
    """
    import matplotlib.pyplot as plt

    # Create parents with complementary building blocks
    n_features = 30

    parent1 = BinaryIndividual(np.zeros(n_features, dtype=int))
    parent1.chromosome[0:10] = 1  # Building block 1
    parent1.chromosome[20:25] = 1  # Building block 3

    parent2 = BinaryIndividual(np.zeros(n_features, dtype=int))
    parent2.chromosome[10:20] = 1  # Building block 2
    parent2.chromosome[25:30] = 1  # Building block 4

    # Generate offspring with uniform crossover
    n_offspring = 20
    offspring = []

    for _ in range(n_offspring):
        child1, child2 = uniform_crossover(parent1, parent2, crossover_prob=1.0)
        offspring.append(child1.chromosome)
        offspring.append(child2.chromosome)

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot parents
    axes[0].imshow([parent1.chromosome, parent2.chromosome], cmap='binary', aspect='auto')
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Parent 1', 'Parent 2'])
    axes[0].set_title('Parents with Building Blocks')
    axes[0].axvline(9.5, color='red', linewidth=2)
    axes[0].axvline(19.5, color='red', linewidth=2)
    axes[0].axvline(24.5, color='red', linewidth=2)

    # Plot offspring
    axes[1].imshow(offspring, cmap='binary', aspect='auto')
    axes[1].set_title('Offspring from Uniform Crossover')
    axes[1].set_ylabel('Offspring Index')
    axes[1].axvline(9.5, color='red', linewidth=1, alpha=0.3)
    axes[1].axvline(19.5, color='red', linewidth=1, alpha=0.3)
    axes[1].axvline(24.5, color='red', linewidth=1, alpha=0.3)

    # Plot building block preservation
    block_counts = np.zeros((4, len(offspring)))
    for i, child in enumerate(offspring):
        block_counts[0, i] = np.all(child[0:10] == parent1.chromosome[0:10])
        block_counts[1, i] = np.all(child[10:20] == parent2.chromosome[10:20])
        block_counts[2, i] = np.all(child[20:25] == parent1.chromosome[20:25])
        block_counts[3, i] = np.all(child[25:30] == parent2.chromosome[25:30])

    axes[2].imshow(block_counts, cmap='RdYlGn', aspect='auto')
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['Block 1', 'Block 2', 'Block 3', 'Block 4'])
    axes[2].set_title('Building Block Preservation (green = preserved)')
    axes[2].set_xlabel('Offspring Index')

    plt.tight_layout()
    plt.savefig('building_blocks.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nBuilding Block Preservation:")
    print(f"Block 1 (parent 1): {np.mean(block_counts[0]) * 100:.1f}%")
    print(f"Block 2 (parent 2): {np.mean(block_counts[1]) * 100:.1f}%")
    print(f"Block 3 (parent 1): {np.mean(block_counts[2]) * 100:.1f}%")
    print(f"Block 4 (parent 2): {np.mean(block_counts[3]) * 100:.1f}%")


if __name__ == "__main__":
    analyze_operator_effects()
    test_mutation_rates()
    visualize_building_blocks()
```

## Common Pitfalls

### 1. Mutation Rate Too High

**Problem**: High mutation rate destroys good solutions (becomes random search).

```python
# Bad - mutation rate of 0.5 means 50% of bits flip!
mutant = bit_flip_mutation(individual, mutation_rate=0.5)

# Good - rule of thumb: 1/n
n = len(individual.chromosome)
mutant = bit_flip_mutation(individual, mutation_rate=1/n)
```

**Empirical test**: Average Hamming distance between parent and child should be ~1-2.

### 2. Always Applying Crossover

**Problem**: No clones preserved, loses good solutions.

```python
# Bad - always crossover
child1, child2 = uniform_crossover(p1, p2, crossover_prob=1.0)

# Good - probabilistic application
child1, child2 = uniform_crossover(p1, p2, crossover_prob=0.8)
# 20% of time, returns clones of parents
```

### 3. Wrong Crossover for Problem Structure

**Problem**: Using single-point for unordered features.

```python
# Bad for feature selection (positional bias)
child1, child2 = single_point_crossover(p1, p2)

# Good - no positional assumptions
child1, child2 = uniform_crossover(p1, p2)
```

### 4. Not Enforcing Constraints After Mutation

```python
# Bad - might create invalid solution
def bad_mutation(individual):
    mutant = individual.copy()
    for i in range(len(mutant.chromosome)):
        if np.random.random() < 0.01:
            mutant.chromosome[i] = 1 - mutant.chromosome[i]
    return mutant  # Might have 0 features!

# Good - enforce constraints
def good_mutation(individual, min_features=1):
    mutant = bit_flip_mutation(individual, mutation_rate=0.01)
    # Constraint enforcement happens inside bit_flip_mutation
    return mutant
```

## Connections

### Prerequisites
- Encoding strategies (binary, integer)
- Selection operators
- Basic probability

### Leads To
- Replacement strategies
- Parameter tuning
- Adaptive operators
- Multi-objective optimization

### Related Concepts
- Schema theorem and building blocks
- Linkage learning
- Island models (migration as crossover-like operator)
- Memetic algorithms (local search as "smart mutation")

## Practice Problems

### Problem 1: Parameterized Crossover

Implement a crossover operator that interpolates between single-point and uniform.

```python
def hybrid_crossover(parent1: Individual, parent2: Individual,
                    alpha: float = 0.5) -> Tuple[Individual, Individual]:
    """
    Hybrid crossover.

    alpha = 0.0 → single-point crossover
    alpha = 1.0 → uniform crossover
    alpha = 0.5 → blend

    Hint: With probability alpha, use uniform; otherwise single-point.
    Or: Use uniform with swap_prob = alpha.
    """
    pass
```

### Problem 2: Smart Mutation

Implement mutation that's less likely to flip important features.

```python
def feature_importance_mutation(individual: Individual,
                                feature_importances: np.ndarray,
                                base_rate: float = 0.01) -> Individual:
    """
    Mutation rate inversely proportional to feature importance.

    Less important features mutated more frequently.
    Helps preserve good building blocks.
    """
    pass
```

### Problem 3: Comparison on Test Problem

Compare crossover operators on a synthetic feature selection problem.

```python
def compare_crossover_operators():
    """
    Create a test problem where:
    - 4 groups of 5 highly correlated features
    - Optimal solution: 1 feature from each group

    Test which crossover operator finds solution fastest.
    Measure: generations to convergence, success rate
    """
    pass
```

### Problem 4: Self-Adaptive Operators

Implement operators that adapt their parameters based on fitness improvement.

```python
class AdaptiveOperatorSet:
    """
    Maintains multiple operators and adapts their usage based on success.

    Track which operators produce fit offspring, use them more.
    """

    def __init__(self):
        self.operators = [single_point_crossover, two_point_crossover, uniform_crossover]
        self.success_counts = [0, 0, 0]
        self.usage_counts = [0, 0, 0]

    def select_operator(self) -> Callable:
        """Select operator based on success rates."""
        pass

    def update_success(self, operator_idx: int, improved: bool):
        """Update statistics when offspring evaluated."""
        pass
```

### Problem 5: Linkage-Learning Crossover

Implement crossover that learns which features should stay together.

```python
def ltga_crossover(parent1: Individual, parent2: Individual,
                  linkage_groups: List[List[int]]) -> Tuple[Individual, Individual]:
    """
    Linkage-Tree Genetic Algorithm crossover.

    Swaps entire linkage groups (features that work well together)
    rather than individual features.

    Parameters
    ----------
    linkage_groups : List[List[int]]
        Groups of feature indices that should be kept together
        Example: [[0,1,2], [3,4], [5,6,7,8]]

    Hint: For each group, decide randomly whether to inherit
    from parent1 or parent2, but keep the group intact.
    """
    pass
```

## Further Reading

### Academic Papers

- De Jong, K.A., & Spears, W.M. (1992). "A formal analysis of the role of multi-point crossover in genetic algorithms". Annals of Mathematics and Artificial Intelligence, 5(1), 1-26.
  - Theoretical analysis of crossover operators

- Spears, W.M., & De Jong, K.A. (1991). "On the virtues of parameterized uniform crossover". Naval Research Laboratory.
  - Shows uniform crossover advantages for feature selection

- Syswerda, G. (1989). "Uniform crossover in genetic algorithms". ICGA, 89, 2-9.
  - Introduction of uniform crossover

- Bäck, T. (1992). "Self-adaptation in genetic algorithms". EuroConf. Parallel Problem Solving from Nature, 263-271.
  - Adaptive mutation rates

### Books

- Eiben, A.E., & Smith, J.E. (2015). "Introduction to Evolutionary Computing" (2nd ed.)
  - Chapter 5: Variation operators

- Mitchell, M. (1998). "An Introduction to Genetic Algorithms"
  - Chapter 4: Theoretical foundations (Schema Theorem)

### Implementation Resources

- DEAP Variation Operators: https://deap.readthedocs.io/en/master/api/tools.html#crossover
- Genetic Algorithm Parameter Guide: http://www.geatbx.com/docu/algindex-02.html#P300_19505

### Key Empirical Findings

1. **Uniform crossover typically best for feature selection** (no positional structure)
2. **Mutation rate of 1/n is robust** across many problems
3. **Crossover rate 0.6-0.9** is standard (0.8 is common default)
4. **Building block preservation** matters: use crossover that respects problem structure
5. **Adaptive rates** can improve performance, especially adaptive mutation

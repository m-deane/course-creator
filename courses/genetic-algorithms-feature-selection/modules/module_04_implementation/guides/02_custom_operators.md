# Custom Genetic Operators: Crossover, Mutation, and Selection

> **Reading time:** ~13 min | **Module:** 4 — Implementation | **Prerequisites:** 01 DEAP Implementation

## In Brief

Custom genetic operators adapt the GA to problem-specific structure, improving convergence speed and solution quality. Standard operators (uniform crossover, bit-flip mutation, tournament selection) work broadly but ignore domain knowledge. Custom operators encode problem constraints, exploit feature relationships, and bias search toward promising regions while maintaining population diversity.

<div class="callout-insight">
Generic GAs treat all genes equally, but features have structure: feature groups (categorical expansions), hierarchical dependencies (interaction terms require base features), and domain constraints (budget limits, correlation requirements). Custom operators that respect this structure converge 2-5× faster than standard operators. The key: design operators that produce valid, high-quality offspring while maintaining diversity.
</div>


![Crossover Types](./crossover_types.svg)

![Mutation Types](./mutation_types.svg)

## Formal Definition

### Crossover Operators

**Purpose:** Combine parent solutions to create offspring

**Standard Uniform Crossover:**
$$\text{child}_i = \begin{cases}
\text{parent}_1[i] & \text{with probability } 0.5 \\
\text{parent}_2[i] & \text{with probability } 0.5
\end{cases}$$

**Custom Feature-Group-Aware Crossover:**
- Swap entire feature groups (e.g., all one-hot encoded categories)
- Preserve within-group dependencies
- Reduce invalid feature combinations

**Mathematical Formulation:**

Let features be partitioned into groups $G_1, G_2, ..., G_k$

$$\text{child}[G_j] = \begin{cases}
\text{parent}_1[G_j] & \text{with probability } 0.5 \\
\text{parent}_2[G_j] & \text{with probability } 0.5
\end{cases}$$

### Mutation Operators

**Purpose:** Introduce random variation, explore search space

**Standard Bit-Flip:**
$$\text{gene}_i = 1 - \text{gene}_i \quad \text{with probability } p_m$$

**Custom Guided Mutation:**
- Feature importance-weighted mutation (higher $p_m$ for important features)
- Correlation-aware mutation (flip correlated features together)
- Budget-preserving mutation (swap selected/unselected to maintain count)

**Adaptive Mutation Rate:**
$$p_m(t) = p_{\max} \cdot \left(1 - \frac{t}{T_{\max}}\right)^\alpha + p_{\min}$$

where $t$ is current generation, $T_{\max}$ is max generations.

### Selection Operators

**Purpose:** Choose parents for reproduction

**Fitness-Proportional Selection:**
$$P(\text{select}_i) = \frac{f_i}{\sum_{j=1}^N f_j}$$

**Custom Diversity-Preserving Selection:**
- Maintain population diversity via fitness sharing
- Penalize similar solutions to encourage exploration

**Fitness Sharing:**
$$f_i' = \frac{f_i}{\sum_{j=1}^N \text{similarity}(i, j)}$$

where similarity decreases effective fitness of crowded solutions.

## Intuitive Explanation

### Why Custom Operators Matter

**Problem: One-Hot Encoded Features**

Standard dataset:
```
Features: [city_NYC, city_LA, city_CHI, income, age]
```

Valid solution: Exactly one city selected
```
[1, 0, 0, 1, 1] ✓ (NYC, income, age)
```

**Standard Crossover Problem:**
```
Parent 1: [1, 0, 0, 1, 1]  (NYC)
Parent 2: [0, 1, 0, 1, 0]  (LA)

Uniform crossover might produce:
Child:    [1, 1, 0, 1, 1]  ✗ (Both NYC and LA selected!)
```

**Custom Group-Aware Crossover:**
```
Groups: {city_features: [0,1,2], other: [3,4]}

Crossover swaps entire city group:
Child:    [0, 1, 0, 1, 1]  ✓ (LA, income, age)
```

### Feature Importance-Weighted Mutation

**Scenario:** 100 features, 10 are highly predictive

**Standard Mutation:**
- All features have $p_m = 0.01$
- Important features changed as often as noise features
- Slow convergence

**Weighted Mutation:**
- Important features: $p_m = 0.05$ (higher exploration)
- Noise features: $p_m = 0.02$ (lower exploration)
- Focus search on promising features

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Feature importances from random forest
importances = [0.15, 0.12, 0.10, 0.05, 0.03, ..., 0.001]

# Mutation rates proportional to importance
mutation_rates = 0.01 + 0.04 * (importances / max(importances))
```
</div>


### Correlation-Aware Mutation

**Problem:** Highly correlated features should be treated together

```
Features: [GDP, GDP_per_capita, population]
GDP = GDP_per_capita × population
```

If GDP is selected, likely want GDP_per_capita as well.

**Standard Mutation:** Flips independently
**Correlation-Aware:** Flips correlated features together with higher probability

## Code Implementation

### Feature-Group-Aware Crossover

```python
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class FeatureGroup:
    """Defines a group of related features."""
    name: str
    indices: List[int]
    group_type: str  # 'one_hot', 'interaction', 'polynomial', 'independent'

class CustomCrossover:
    """
    Custom crossover operators for feature selection.
    """

    def __init__(self, feature_groups: List[FeatureGroup]):
        self.feature_groups = feature_groups

    def group_aware_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        crossover_prob: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover that respects feature groups.

        For one-hot groups, swap entire group to maintain validity.
        For other groups, allow within-group mixing.
        """
        if np.random.random() > crossover_prob:
            return parent1.copy(), parent2.copy()

        child1 = parent1.copy()
        child2 = parent2.copy()

        for group in self.feature_groups:
            if group.group_type == 'one_hot':
                # Swap entire group
                if np.random.random() < 0.5:
                    child1[group.indices] = parent2[group.indices]
                    child2[group.indices] = parent1[group.indices]
            else:
                # Standard uniform crossover within group
                for idx in group.indices:
                    if np.random.random() < 0.5:
                        child1[idx], child2[idx] = parent2[idx], parent1[idx]

        return child1, child2

    def hierarchical_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        interaction_map: Dict[int, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover that preserves hierarchical dependencies.

        If base feature X is selected, interaction X*Y can be selected.
        If X is not selected, X*Y must not be selected.

        Args:
            interaction_map: Maps interaction feature index to base feature indices
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Standard crossover first
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent2, parent1)
        child2 = np.where(mask, parent1, parent2)

        # Fix hierarchical violations
        for interaction_idx, base_indices in interaction_map.items():
            # If any base feature is 0, interaction must be 0
            for child in [child1, child2]:
                if any(child[base_idx] == 0 for base_idx in base_indices):
                    child[interaction_idx] = 0

        return child1, child2


# Example usage
print("=" * 70)
print("CUSTOM CROSSOVER OPERATORS")
print("=" * 70)

# Define feature groups
feature_groups = [
    FeatureGroup(name='city', indices=[0, 1, 2], group_type='one_hot'),
    FeatureGroup(name='product', indices=[3, 4, 5, 6], group_type='one_hot'),
    FeatureGroup(name='numeric', indices=[7, 8, 9], group_type='independent')
]

crossover = CustomCrossover(feature_groups)

# Parents
parent1 = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0])  # NYC, product_B, num1, num2
parent2 = np.array([0, 1, 0, 1, 0, 0, 0, 0, 1, 1])  # LA, product_A, num2, num3

print("\nParents:")
print(f"Parent 1: {parent1}")
print(f"Parent 2: {parent2}")

# Standard uniform crossover (can violate constraints)
child1_std = np.where(np.random.random(10) < 0.5, parent1, parent2)
child2_std = np.where(np.random.random(10) < 0.5, parent2, parent1)

print("\nStandard Uniform Crossover:")
print(f"Child 1: {child1_std}")
print(f"  City selected: {child1_std[:3].sum()} (should be 1)")
print(f"  Product selected: {child1_std[3:7].sum()} (should be 1)")

# Group-aware crossover
child1_custom, child2_custom = crossover.group_aware_crossover(parent1, parent2)

print("\nGroup-Aware Crossover:")
print(f"Child 1: {child1_custom}")
print(f"  City selected: {child1_custom[:3].sum()} (should be 1)")
print(f"  Product selected: {child1_custom[3:7].sum()} (should be 1)")

# Hierarchical crossover example
print("\n" + "=" * 70)
print("HIERARCHICAL CROSSOVER")
print("=" * 70)

# Features: [X1, X2, X3, X1*X2, X1*X3, X2*X3]
interaction_map = {
    3: [0, 1],  # X1*X2 requires X1 and X2
    4: [0, 2],  # X1*X3 requires X1 and X3
    5: [1, 2]   # X2*X3 requires X2 and X3
}

parent1_hier = np.array([1, 1, 0, 1, 0, 0])  # X1, X2, X1*X2 selected
parent2_hier = np.array([1, 0, 1, 0, 1, 0])  # X1, X3, X1*X3 selected

print(f"\nParent 1: {parent1_hier} (X1, X2, X1*X2)")
print(f"Parent 2: {parent2_hier} (X1, X3, X1*X3)")

child1_hier, child2_hier = crossover.hierarchical_crossover(
    parent1_hier,
    parent2_hier,
    interaction_map
)

print(f"\nChild 1: {child1_hier}")
print(f"  X1*X2 valid: {child1_hier[3] == 0 or (child1_hier[0] == 1 and child1_hier[1] == 1)}")
print(f"  X1*X3 valid: {child1_hier[4] == 0 or (child1_hier[0] == 1 and child1_hier[2] == 1)}")
```

### Custom Mutation Operators

```python
class CustomMutation:
    """
    Custom mutation operators for feature selection.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features

    def importance_weighted_mutation(
        self,
        individual: np.ndarray,
        feature_importances: np.ndarray,
        base_rate: float = 0.01,
        importance_weight: float = 3.0
    ) -> np.ndarray:
        """
        Mutation with higher rates for important features.

        Args:
            individual: Chromosome
            feature_importances: Importance scores [0, 1]
            base_rate: Minimum mutation rate
            importance_weight: Multiplier for important features
        """
        mutant = individual.copy()

        # Scale importances
        importances_scaled = feature_importances / (feature_importances.max() + 1e-10)

        # Compute mutation rates
        mutation_rates = base_rate + (importance_weight * base_rate) * importances_scaled

        # Mutate
        for i in range(len(mutant)):
            if np.random.random() < mutation_rates[i]:
                mutant[i] = 1 - mutant[i]

        # Ensure at least one feature
        if mutant.sum() == 0:
            mutant[np.argmax(feature_importances)] = 1

        return mutant

    def correlation_aware_mutation(
        self,
        individual: np.ndarray,
        correlation_matrix: np.ndarray,
        base_rate: float = 0.01,
        correlation_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Mutate correlated features together.

        If feature i is mutated and |corr(i,j)| > threshold,
        also mutate feature j with high probability.
        """
        mutant = individual.copy()

        for i in range(len(mutant)):
            if np.random.random() < base_rate:
                # Flip this feature
                mutant[i] = 1 - mutant[i]

                # Find highly correlated features
                high_corr_indices = np.where(
                    np.abs(correlation_matrix[i]) > correlation_threshold
                )[0]

                # Flip correlated features with some probability
                for j in high_corr_indices:
                    if j != i and np.random.random() < 0.5:
                        mutant[j] = 1 - mutant[j]

        # Ensure at least one feature
        if mutant.sum() == 0:
            mutant[np.random.randint(len(mutant))] = 1

        return mutant

    def swap_mutation(
        self,
        individual: np.ndarray,
        n_swaps: int = 1
    ) -> np.ndarray:
        """
        Budget-preserving mutation: swap selected/unselected features.

        Maintains constant number of selected features.
        """
        mutant = individual.copy()

        selected = np.where(mutant == 1)[0]
        unselected = np.where(mutant == 0)[0]

        for _ in range(n_swaps):
            if len(selected) > 0 and len(unselected) > 0:
                # Pick one from each set
                turn_off = np.random.choice(selected)
                turn_on = np.random.choice(unselected)

                # Swap
                mutant[turn_off] = 0
                mutant[turn_on] = 1

                # Update sets
                selected = np.where(mutant == 1)[0]
                unselected = np.where(mutant == 0)[0]

        return mutant

    def adaptive_mutation(
        self,
        individual: np.ndarray,
        generation: int,
        max_generations: int,
        initial_rate: float = 0.1,
        final_rate: float = 0.01
    ) -> np.ndarray:
        """
        Mutation rate decreases over generations.

        High exploration early, fine-tuning later.
        """
        # Linear decay
        progress = generation / max_generations
        mutation_rate = initial_rate - progress * (initial_rate - final_rate)

        mutant = individual.copy()
        for i in range(len(mutant)):
            if np.random.random() < mutation_rate:
                mutant[i] = 1 - mutant[i]

        # Ensure at least one feature
        if mutant.sum() == 0:
            mutant[np.random.randint(len(mutant))] = 1

        return mutant


# Example usage
print("\n" + "=" * 70)
print("CUSTOM MUTATION OPERATORS")
print("=" * 70)

n_features = 10
individual = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])

mutation = CustomMutation(n_features)

# Importance-weighted mutation
feature_importances = np.array([0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])

print("\nImportance-Weighted Mutation:")
print(f"Original: {individual}")
print(f"Importances: {feature_importances}")

mutant_importance = mutation.importance_weighted_mutation(
    individual,
    feature_importances,
    base_rate=0.05
)
print(f"Mutated: {mutant_importance}")

# Correlation-aware mutation
correlation_matrix = np.random.randn(n_features, n_features)
correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Symmetric
np.fill_diagonal(correlation_matrix, 1.0)

print("\n" + "-" * 70)
print("Correlation-Aware Mutation:")
print(f"Original: {individual}")

mutant_corr = mutation.correlation_aware_mutation(
    individual,
    correlation_matrix,
    base_rate=0.1
)
print(f"Mutated: {mutant_corr}")

# Swap mutation (budget-preserving)
print("\n" + "-" * 70)
print("Swap Mutation (Budget-Preserving):")
print(f"Original: {individual} (count={individual.sum()})")

mutant_swap = mutation.swap_mutation(individual, n_swaps=2)
print(f"Mutated: {mutant_swap} (count={mutant_swap.sum()})")

# Adaptive mutation
print("\n" + "-" * 70)
print("Adaptive Mutation (Over Generations):")

for gen in [0, 25, 50, 75, 100]:
    mutant_adaptive = mutation.adaptive_mutation(
        individual,
        generation=gen,
        max_generations=100
    )
    progress = gen / 100
    expected_rate = 0.1 - progress * (0.1 - 0.01)
    print(f"  Gen {gen:3d}: Expected rate={expected_rate:.3f}")
```

### Custom Selection Operators

```python
class CustomSelection:
    """
    Custom selection operators.
    """

    def diversity_preserving_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: np.ndarray,
        n_select: int,
        diversity_weight: float = 0.3
    ) -> List[np.ndarray]:
        """
        Select individuals balancing fitness and diversity.

        Uses fitness sharing to penalize crowded solutions.
        """
        n_pop = len(population)

        # Compute pairwise distances (Hamming distance)
        distances = np.zeros((n_pop, n_pop))
        for i in range(n_pop):
            for j in range(i+1, n_pop):
                dist = np.sum(population[i] != population[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Fitness sharing
        shared_fitness = np.zeros(n_pop)
        for i in range(n_pop):
            # Count similar individuals
            similarity = np.exp(-distances[i] / len(population[0]))  # Exponential similarity
            sharing_factor = similarity.sum()

            # Adjust fitness
            shared_fitness[i] = fitness_scores[i] / sharing_factor

        # Weighted combination
        combined_scores = (1 - diversity_weight) * fitness_scores + diversity_weight * shared_fitness

        # Select top individuals
        selected_indices = np.argsort(combined_scores)[-n_select:]
        return [population[i] for i in selected_indices]

    def niching_selection(
        self,
        population: List[np.ndarray],
        fitness_scores: np.ndarray,
        n_niches: int = 5
    ) -> List[np.ndarray]:
        """
        Select best individual from each niche.

        Maintains population diversity by ensuring all niches represented.
        """
        # Cluster population into niches (simple k-means on chromosomes)
        from sklearn.cluster import KMeans

        pop_array = np.array(population)
        kmeans = KMeans(n_clusters=n_niches, random_state=42, n_init=10)
        niche_labels = kmeans.fit_predict(pop_array)

        # Select best from each niche
        selected = []
        for niche_id in range(n_niches):
            niche_mask = niche_labels == niche_id
            if niche_mask.sum() > 0:
                niche_fitness = fitness_scores[niche_mask]
                best_in_niche_idx = np.where(niche_mask)[0][niche_fitness.argmax()]
                selected.append(population[best_in_niche_idx])

        return selected


# Example usage
print("\n" + "=" * 70)
print("CUSTOM SELECTION OPERATORS")
print("=" * 70)

# Create diverse population
np.random.seed(42)
population = [np.random.randint(0, 2, 10) for _ in range(20)]
fitness_scores = np.random.randn(20) + 5  # Random fitness

print(f"\nPopulation size: {len(population)}")
print(f"Fitness range: [{fitness_scores.min():.2f}, {fitness_scores.max():.2f}]")

selection = CustomSelection()

# Diversity-preserving selection
selected_diverse = selection.diversity_preserving_selection(
    population,
    fitness_scores,
    n_select=5,
    diversity_weight=0.3
)

print("\nDiversity-Preserving Selection (5 selected):")
for i, ind in enumerate(selected_diverse):
    print(f"  {i+1}. {ind}")

# Compare with top-5 by fitness only
top5_indices = np.argsort(fitness_scores)[-5:]
print("\nTop 5 by Fitness Only:")
for i, idx in enumerate(top5_indices):
    print(f"  {i+1}. {population[idx]}")

# Compute diversity
def population_diversity(pop):
    """Average Hamming distance between all pairs."""
    n = len(pop)
    total_dist = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_dist += np.sum(pop[i] != pop[j])
            count += 1
    return total_dist / count if count > 0 else 0

div_diverse = population_diversity(selected_diverse)
div_fitness = population_diversity([population[i] for i in top5_indices])

print(f"\nPopulation Diversity:")
print(f"  Diversity-preserving: {div_diverse:.2f}")
print(f"  Fitness-only: {div_fitness:.2f}")
```

## Common Pitfalls

<div class="callout-warning">

⚠️ **Warning:** The most frequent mistake with custom operators is producing invalid offspring (e.g., selecting two mutually exclusive one-hot features). Always add constraint validation after every crossover and mutation call.

</div>

**1. Violating Problem Constraints**
- Problem: Custom operators produce invalid solutions (e.g., two cities selected)
- Symptom: Fitness function must constantly repair individuals
- Solution: Design operators to maintain validity, add constraint checks

**2. Over-Complicating Operators**
- Problem: Custom operator has 10 parameters, hard to tune
- Symptom: Performance worse than standard operators
- Solution: Start simple, add complexity only if justified by performance

**3. Ignoring Computational Cost**
- Problem: Correlation-aware mutation computes full correlation matrix every call
- Symptom: GA runs 10× slower
- Solution: Precompute and cache expensive operations

**4. Breaking Genetic Algorithm Theory**
- Problem: Custom mutation always improves fitness (greedy)
- Symptom: Loss of exploration, premature convergence
- Solution: Maintain randomness, don't always improve

**5. Not Testing Operator Effectiveness**
- Problem: Assume custom operator is better without evidence
- Symptom: Worse performance than standard GA
- Solution: A/B test custom vs standard operators on validation set

## Connections

<div class="callout-info">
ℹ️ **How this connects to the rest of the course:**
</div>

**Builds on:**
- Module 1: GA fundamentals (standard operators)
- Module 2: Fitness functions (operators interact with fitness)
- Domain knowledge (problem structure informs custom operators)

**Leads to:**
- Module 4.3: Production considerations (efficient operator implementation)
- Module 5: Advanced methods (adaptive operators, hybrid approaches)
- Hyperparameter optimization (tuning custom operator parameters)

**Related concepts:**
- Constraint satisfaction (operators respecting constraints)
- Local search (mutation as local neighborhood exploration)
- Problem-specific heuristics (domain knowledge integration)

## Practice Problems

1. **Operator Design**
   Feature selection problem with 3 feature groups: demographics (5), financials (10), behavioral (15).
   Design crossover that: (a) never mixes groups, (b) allows within-group mixing.
   What crossover probability per group?

2. **Mutation Rate Calculation**
   100 features, importance-weighted mutation.
   Top 10 features have importance 0.10-0.15.
   Bottom 50 features have importance < 0.01.
   What mutation rates to achieve: average 2 flips per mutation?

3. **Hierarchical Dependencies**
   Base features: [A, B, C]
   Interactions: [A*B, A*C, B*C, A*B*C]
   Design mutation that preserves: "If A*B selected, then A and B selected."

4. **Diversity Analysis**
   Population of 50, each chromosome length 20.
   Average Hamming distance: 8.5.
   Is diversity high or low?
   What diversity target for effective search?

5. **Swap Mutation Budget**
   Start with 10 features selected.
   Swap mutation: n_swaps=3.
   After mutation, how many features selected?
   What if n_swaps > number of selected features?

## Further Reading

**Genetic Operator Design:**
1. **"Introduction to Evolutionary Computing" by Eiben & Smith** - Operator theory
2. **"Genetic Algorithms in Search, Optimization, and Machine Learning" by Goldberg** - Classical GA operators
3. **"Problem-Specific Operators in Genetic Algorithms"** - Custom operator design

**Feature Selection Specific:**
4. **"A Survey on Feature Selection using Genetic Algorithms"** - FS operator survey
5. **"Handling Constraints in Genetic Algorithms"** - Constraint-preserving operators
6. **"Group Genetic Algorithms"** - Group-aware operators

**Advanced Topics:**
7. **"Self-Adaptive Genetic Algorithms"** - Adaptive operator parameters
8. **"Diversity Preservation in Genetic Algorithms"** - Niching, fitness sharing
9. **"Hybrid Genetic Algorithms"** - Combining GA with local search

**Implementation:**
10. **"Efficient Implementation of Genetic Algorithms"** - Performance optimization
11. **"DEAP Documentation"** - Python GA framework with custom operators
12. **"GA Operator Benchmarking"** - Testing operator effectiveness

---

*"Standard operators are general. Custom operators are optimal. Choose based on problem structure."*
---

**Next:** [Companion Slides](./02_custom_operators_slides.md) | [Notebook](../notebooks/02_custom_operators.ipynb)

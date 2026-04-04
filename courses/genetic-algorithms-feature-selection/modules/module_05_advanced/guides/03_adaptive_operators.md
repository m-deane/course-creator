# Adaptive Genetic Operators: Self-Adaptive Parameters and Mutation Rates

> **Reading time:** ~11 min | **Module:** 5 — Advanced Topics | **Prerequisites:** 02 Hybrid Methods

## In Brief

Adaptive operators automatically adjust GA parameters (mutation rate, crossover probability, population diversity) during evolution based on search progress, eliminating manual tuning and improving robustness. Fixed parameters work well for specific problems but fail across diverse landscapes. Self-adaptive GAs evolve their own parameters alongside solutions, with successful parameter settings propagating through the population via natural selection, achieving performance comparable to expert-tuned parameters without human intervention.

<div class="callout-insight">
Optimal GA parameters change during search: early exploration needs high mutation/diversity, late exploitation needs low mutation/focused search. Fixed parameters are either too exploratory (slow convergence) or too exploitative (premature convergence). Adaptive operators measure population state (diversity, fitness improvement, convergence rate) and dynamically adjust parameters. Self-adaptive operators encode parameters in chromosomes, allowing evolution to discover optimal settings for the current search phase.
</div>


![Fitness Landscape](./fitness_landscape.svg)

## Formal Definition

### Adaptive Parameter Control

**Deterministic Adaptation (Scheduled):**
$$p_m(t) = p_{\max} - \frac{(p_{\max} - p_{\min}) \cdot t}{T}$$

Linear decrease from $p_{\max}$ to $p_{\min}$ over $T$ generations.

**Feedback-Based Adaptation:**
$$p_m(t+1) = \begin{cases}
p_m(t) \cdot (1 + \alpha) & \text{if improvement stagnates} \\
p_m(t) \cdot (1 - \beta) & \text{if improvement rapid} \\
p_m(t) & \text{otherwise}
\end{cases}$$

where $\alpha, \beta$ control adaptation speed.

**Diversity-Based Adaptation:**

Population diversity:
$$D(t) = \frac{1}{P(P-1)} \sum_{i=1}^P \sum_{j=i+1}^P \text{distance}(x_i, x_j)$$

Adaptation rule:
$$p_m(t) = p_{\min} + (p_{\max} - p_{\min}) \cdot \left(1 - \frac{D(t)}{D_{\max}}\right)$$

High diversity → Low mutation (exploitation)
Low diversity → High mutation (re-exploration)

### Self-Adaptive Parameters

**Extended Chromosome:**
$$\text{Individual} = (x_1, ..., x_n, p_m, p_c, \sigma)$$

where:
- $x_1, ..., x_n$: Feature selection bits
- $p_m$: Individual's mutation rate
- $p_c$: Individual's crossover probability
- $\sigma$: Mutation strength (for continuous params)

**Parameter Evolution:**
1. Mutate parameters with meta-mutation:
   $$p_m' = p_m \cdot e^{\tau \cdot N(0,1)}$$
   where $\tau = \frac{1}{\sqrt{n}}$ (learning rate)

2. Apply mutated parameters to solution:
   $$x' = \text{mutate}(x, p_m')$$

3. Selection acts on both solution quality and parameter effectiveness

**Success-Based Adaptation (1/5 Rule):**

Track success rate over window:
$$\text{SR}(t) = \frac{\text{Improvements in last } w \text{ generations}}{w}$$

Adaptation:
$$p_m(t+1) = \begin{cases}
p_m(t) / c & \text{if SR} > 1/5 \text{ (too successful)} \\
p_m(t) \cdot c & \text{if SR} < 1/5 \text{ (too few successes)} \\
p_m(t) & \text{if SR} \approx 1/5
\end{cases}$$

### Multi-Population Adaptive GAs

**Island Model with Parameter Diversity:**

Maintain $k$ sub-populations with different parameter settings:
$$\text{Island}_i: (P_i, p_m^i, p_c^i)$$

Periodic migration:
- Best individuals migrate between islands
- Parameter settings implicitly compete
- Successful parameters spread to other islands

## Intuitive Explanation

### Why Fixed Parameters Fail

**Scenario: Feature Selection with 100 features**

**Fixed Mutation Rate = 0.01:**
```
Generation 0:    [Exploring broadly, 0.01 works well]
Generation 20:   [Converging, 0.01 still okay]
Generation 40:   [Near optimum, 0.01 disrupts good solutions]
Generation 50:   [Stuck, can't fine-tune]
```

Result: Converges to suboptimal solution because late-stage exploration too coarse.

**Fixed Mutation Rate = 0.001:**
```
Generation 0:    [Too little exploration, slow progress]
Generation 20:   [Still exploring slowly...]
Generation 40:   [Still exploring...]
Generation 50:   [Never converged, but making progress]
```

Result: Never reaches good solutions within budget.

**Adaptive Mutation:**
```
Generation 0:    [Start at 0.05, rapid exploration]
Generation 20:   [Reduce to 0.02, focused search]
Generation 40:   [Reduce to 0.005, fine-tuning]
Generation 50:   [0.001, local refinement]
```

Result: Fast initial progress, smooth convergence to optimum.

### Self-Adaptive Example

**Individual with High Mutation Rate:**
```
Chromosome: [1,0,1,1,0]
Parameters: pm=0.10 (high)
Offspring:  [1,1,0,1,1] (many changes)
Fitness:    Worse (too disruptive)
→ Dies out in selection
→ High mutation rate disappears
```

**Individual with Moderate Mutation Rate:**
```
Chromosome: [1,0,1,1,0]
Parameters: pm=0.02 (moderate)
Offspring:  [1,0,1,0,0] (small changes)
Fitness:    Better (incremental improvement)
→ Survives selection
→ Moderate mutation rate spreads
```

Evolution discovers: "At this search stage, moderate mutation is optimal."

### Diversity-Based Adaptation in Action

**High Diversity (Early Search):**
```
Population: [Very different individuals]
Diversity: 0.8 (high)
Decision: Reduce mutation to 0.01 (let crossover work)
```

**Low Diversity (Premature Convergence):**
```
Population: [Very similar individuals]
Diversity: 0.2 (low)
Decision: Increase mutation to 0.08 (restore exploration)
```

## Code Implementation

### Adaptive Mutation Rate

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">__init__.py</span>
</div>

```python
import numpy as np
from typing import List, Callable
from collections import deque

class AdaptiveGA:
    """
    Genetic algorithm with adaptive mutation rate.
    """

    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        population_size: int = 50,
        n_generations: int = 50,
        adaptation_type: str = 'diversity',  # 'linear', 'feedback', 'diversity'
        initial_mutation_rate: float = 0.05,
        min_mutation_rate: float = 0.001,
        max_mutation_rate: float = 0.1,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.adaptation_type = adaptation_type
        self.mutation_rate = initial_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.rng = np.random.RandomState(random_state)

        # Tracking
        self.mutation_rate_history = []
        self.diversity_history = []
        self.improvement_history = deque(maxlen=5)  # Last 5 generations

    def _compute_diversity(self, population: List[np.ndarray]) -> float:
        """
        Compute population diversity (average Hamming distance).
        """
        n_pop = len(population)
        total_distance = 0
        count = 0

        for i in range(n_pop):
            for j in range(i + 1, n_pop):
                total_distance += np.sum(population[i] != population[j])
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        normalized_diversity = avg_distance / self.n_features  # Normalize to [0, 1]

        return normalized_diversity

    def _adapt_mutation_rate(
        self,
        generation: int,
        population: List[np.ndarray],
        best_fitness: float
    ):
        """
        Adapt mutation rate based on strategy.
        """
        if self.adaptation_type == 'linear':
            # Linear decrease
            progress = generation / self.n_generations
            self.mutation_rate = (
                self.max_mutation_rate -
                progress * (self.max_mutation_rate - self.min_mutation_rate)
            )

        elif self.adaptation_type == 'diversity':
            # Diversity-based
            diversity = self._compute_diversity(population)
            self.diversity_history.append(diversity)

            # Inverse relationship: low diversity → high mutation
            self.mutation_rate = (
                self.min_mutation_rate +
                (self.max_mutation_rate - self.min_mutation_rate) * (1 - diversity)
            )

        elif self.adaptation_type == 'feedback':
            # Feedback-based (improvement rate)
            self.improvement_history.append(best_fitness)

            if len(self.improvement_history) >= 5:
                recent_improvement = (
                    self.improvement_history[-1] - self.improvement_history[0]
                )

                if recent_improvement < 0.001:  # Stagnation
                    self.mutation_rate = min(
                        self.mutation_rate * 1.2,
                        self.max_mutation_rate
                    )
                elif recent_improvement > 0.01:  # Rapid improvement
                    self.mutation_rate = max(
                        self.mutation_rate * 0.9,
                        self.min_mutation_rate
                    )

        # Clip to bounds
        self.mutation_rate = np.clip(
            self.mutation_rate,
            self.min_mutation_rate,
            self.max_mutation_rate
        )

        self.mutation_rate_history.append(self.mutation_rate)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutate with current adaptive rate."""
        mutant = individual.copy()

        for i in range(self.n_features):
            if self.rng.random() < self.mutation_rate:
                mutant[i] = 1 - mutant[i]

        if mutant.sum() == 0:
            mutant[self.rng.randint(self.n_features)] = 1

        return mutant

    def run(self) -> dict:
        """Run adaptive GA."""
        # Initialize population
        population = [
            (self.rng.random(self.n_features) < 0.3).astype(int)
            for _ in range(self.population_size)
        ]

        for ind in population:
            if ind.sum() == 0:
                ind[self.rng.randint(self.n_features)] = 1

        # Evaluate
        fitness_scores = np.array([self.fitness_func(ind) for ind in population])

        best_fitness_history = []

        # Evolution
        for generation in range(self.n_generations):
            best_fitness = fitness_scores.max()
            best_fitness_history.append(best_fitness)

            # Adapt mutation rate
            self._adapt_mutation_rate(generation, population, best_fitness)

            if generation % 10 == 0:
                diversity = self._compute_diversity(population)
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Mutation={self.mutation_rate:.4f}, "
                      f"Diversity={diversity:.3f}")

            # Generate offspring (simplified)
            offspring = []
            for _ in range(self.population_size):
                # Selection (tournament)
                parent_idx = self.rng.choice(
                    self.population_size,
                    3
                )[fitness_scores[self.rng.choice(self.population_size, 3)].argmax()]
                parent = population[parent_idx].copy()

                # Mutation
                child = self._mutate(parent)
                offspring.append(child)

            # Evaluate offspring
            offspring_fitness = np.array([self.fitness_func(ind) for ind in offspring])

            # Elitism
            best_idx = fitness_scores.argmax()
            worst_offspring_idx = offspring_fitness.argmin()
            offspring[worst_offspring_idx] = population[best_idx]
            offspring_fitness[worst_offspring_idx] = fitness_scores[best_idx]

            # Replace
            population = offspring
            fitness_scores = offspring_fitness

        best_idx = fitness_scores.argmax()

        return {
            'best_individual': population[best_idx],
            'best_fitness': fitness_scores[best_idx],
            'best_fitness_history': best_fitness_history,
            'mutation_rate_history': self.mutation_rate_history,
            'diversity_history': self.diversity_history
        }


# Example usage
print("=" * 70)
print("ADAPTIVE MUTATION RATE COMPARISON")
print("=" * 70)

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Generate data
X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=10,
    random_state=42
)

def fitness_function(chromosome):
    if chromosome.sum() == 0:
        return 0
    selected = chromosome.astype(bool)
    X_selected = X[:, selected]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    return cross_val_score(clf, X_selected, y, cv=3).mean()

# Test different adaptation strategies
strategies = ['linear', 'diversity', 'feedback']
results = {}

for strategy in strategies:
    print(f"\n{'-' * 70}")
    print(f"Strategy: {strategy.upper()}")
    print('-' * 70)

    ga = AdaptiveGA(
        n_features=30,
        fitness_func=fitness_function,
        population_size=50,
        n_generations=30,
        adaptation_type=strategy,
        random_state=42
    )

    results[strategy] = ga.run()

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

for strategy, result in results.items():
    print(f"\n{strategy.upper()}:")
    print(f"  Best fitness: {result['best_fitness']:.4f}")
    print(f"  Features selected: {result['best_individual'].sum()}")
    print(f"  Final mutation rate: {result['mutation_rate_history'][-1]:.4f}")
```
</div>


### Self-Adaptive GA

```python
class SelfAdaptiveGA:
    """
    Self-adaptive GA with parameters encoded in chromosomes.
    """

    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        population_size: int = 50,
        n_generations: int = 50,
        tau: float = None,  # Learning rate (default: 1/sqrt(n))
        random_state: int = 42
    ):
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.tau = tau or 1.0 / np.sqrt(n_features)
        self.rng = np.random.RandomState(random_state)

    def _create_individual(self) -> dict:
        """
        Create individual with self-adaptive parameters.

        Returns:
            {
                'chromosome': feature selection bits,
                'mutation_rate': individual mutation rate,
                'crossover_prob': individual crossover probability
            }
        """
        chromosome = (self.rng.random(self.n_features) < 0.3).astype(int)
        if chromosome.sum() == 0:
            chromosome[self.rng.randint(self.n_features)] = 1

        return {
            'chromosome': chromosome,
            'mutation_rate': 0.02 + self.rng.random() * 0.08,  # [0.02, 0.10]
            'crossover_prob': 0.5 + self.rng.random() * 0.4   # [0.5, 0.9]
        }

    def _mutate_parameters(self, individual: dict) -> dict:
        """
        Mutate strategy parameters.
        """
        mutant = individual.copy()

        # Mutate mutation rate (log-normal)
        mutant['mutation_rate'] = individual['mutation_rate'] * np.exp(
            self.tau * self.rng.randn()
        )
        mutant['mutation_rate'] = np.clip(mutant['mutation_rate'], 0.001, 0.2)

        # Mutate crossover probability
        mutant['crossover_prob'] = individual['crossover_prob'] + self.rng.randn() * 0.1
        mutant['crossover_prob'] = np.clip(mutant['crossover_prob'], 0.3, 1.0)

        return mutant

    def _mutate_chromosome(self, individual: dict) -> dict:
        """
        Mutate chromosome using individual's mutation rate.
        """
        mutant = individual.copy()
        chromosome = individual['chromosome'].copy()

        for i in range(self.n_features):
            if self.rng.random() < individual['mutation_rate']:
                chromosome[i] = 1 - chromosome[i]

        if chromosome.sum() == 0:
            chromosome[self.rng.randint(self.n_features)] = 1

        mutant['chromosome'] = chromosome
        return mutant

    def run(self) -> dict:
        """Run self-adaptive GA."""
        # Initialize population
        population = [self._create_individual() for _ in range(self.population_size)]

        # Evaluate
        fitness_scores = np.array([
            self.fitness_func(ind['chromosome']) for ind in population
        ])

        best_fitness_history = []
        avg_mutation_rate_history = []

        # Evolution
        for generation in range(self.n_generations):
            best_fitness = fitness_scores.max()
            best_fitness_history.append(best_fitness)

            # Track average mutation rate
            avg_mutation = np.mean([ind['mutation_rate'] for ind in population])
            avg_mutation_rate_history.append(avg_mutation)

            if generation % 10 == 0:
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Avg Mutation={avg_mutation:.4f}")

            # Generate offspring
            offspring = []
            for _ in range(self.population_size):
                # Select parent
                parent_idx = self.rng.choice(
                    self.population_size,
                    3
                )[fitness_scores[self.rng.choice(self.population_size, 3)].argmax()]
                parent = population[parent_idx]

                # Mutate parameters first
                child = self._mutate_parameters(parent)

                # Then mutate chromosome using new parameters
                child = self._mutate_chromosome(child)

                offspring.append(child)

            # Evaluate
            offspring_fitness = np.array([
                self.fitness_func(ind['chromosome']) for ind in offspring
            ])

            # Elitism
            best_idx = fitness_scores.argmax()
            worst_offspring_idx = offspring_fitness.argmin()
            offspring[worst_offspring_idx] = population[best_idx]
            offspring_fitness[worst_offspring_idx] = fitness_scores[best_idx]

            # Replace
            population = offspring
            fitness_scores = offspring_fitness

        best_idx = fitness_scores.argmax()

        return {
            'best_individual': population[best_idx]['chromosome'],
            'best_fitness': fitness_scores[best_idx],
            'best_mutation_rate': population[best_idx]['mutation_rate'],
            'best_crossover_prob': population[best_idx]['crossover_prob'],
            'best_fitness_history': best_fitness_history,
            'avg_mutation_rate_history': avg_mutation_rate_history
        }


# Example usage
print("\n" + "=" * 70)
print("SELF-ADAPTIVE GA")
print("=" * 70)

sa_ga = SelfAdaptiveGA(
    n_features=30,
    fitness_func=fitness_function,
    population_size=50,
    n_generations=30,
    random_state=42
)

result_self_adaptive = sa_ga.run()

print(f"\nResults:")
print(f"  Best fitness: {result_self_adaptive['best_fitness']:.4f}")
print(f"  Best mutation rate: {result_self_adaptive['best_mutation_rate']:.4f}")
print(f"  Best crossover prob: {result_self_adaptive['best_crossover_prob']:.3f}")
print(f"  Features selected: {result_self_adaptive['best_individual'].sum()}")
```

## Common Pitfalls

<div class="callout-warning">

⚠️ **Warning:** Self-adaptive parameters with weak selection pressure drift randomly rather than converging to useful settings. Ensure tournament size is at least 3 and population is large enough (50+) for parameter evolution to work.

</div>

**1. Over-Adapting**
- Problem: Changing parameters every generation based on noise
- Symptom: Erratic parameter values, unstable search
- Solution: Use moving averages, adapt every N generations

**2. Narrow Adaptation Range**
- Problem: min_mutation = 0.01, max_mutation = 0.015 (too narrow)
- Symptom: Adaptation has no effect
- Solution: Wide range (e.g., 0.001 to 0.1), let adaptation find sweet spot

**3. Self-Adaptive Without Sufficient Selection Pressure**
- Problem: Weak selection, bad parameters survive
- Symptom: Parameters drift randomly, no improvement
- Solution: Strong selection (tournament size 3-5), adequate population

**4. Ignoring Parameter Interactions**
- Problem: Adapting mutation rate while keeping crossover fixed
- Symptom: Suboptimal balance between operators
- Solution: Adapt multiple parameters, or use self-adaptive

**5. No Baseline Comparison**
- Problem: "Adaptive is better!" but never tested vs tuned fixed parameters
- Symptom: Adaptive underperforms good fixed settings
- Solution: Always compare against well-tuned baseline

## Connections

<div class="callout-info">
ℹ️ **How this connects to the rest of the course:**
</div>

**Builds on:**
- Module 1-3: Operators (what to adapt)
- Module 5.2: Hybrid methods (when to apply local search)
- Control theory (feedback loops, stability)

**Leads to:**
- Multi-objective GAs (adaptive Pareto selection)
- Coevolution (parameters coevolve with solutions)
- Meta-learning (learning to learn search strategies)

**Related concepts:**
- Evolution strategies (self-adaptation originated here)
- Reinforcement learning (reward successful parameters)
- Bayesian optimization (adaptive acquisition functions)

## Practice Problems

1. **Diversity Calculation**
   Population: 10 individuals, chromosome length 20.
   Avg Hamming distance: 12.
   What is normalized diversity?
   Is this high or low?

2. **Adaptation Rule Design**
   Design adaptation rule:
   - If no improvement for 5 generations: increase mutation by 50%
   - If improvement every generation: decrease mutation by 10%
   - Max mutation: 0.1, Min: 0.001
   Starting at 0.02, what's the rate after 10 stagnant generations?

3. **Self-Adaptive Convergence**
   Individual A: mutation_rate = 0.10, produces fitness 0.75
   Individual B: mutation_rate = 0.01, produces fitness 0.80
   Which parameter setting survives selection?
   What if B's offspring have fitness 0.75 (less robust)?

4. **Parameter Encoding**
   Encode mutation rate [0.001, 0.1] in 5 bits.
   What resolution?
   Value of binary 10110?
   Is 5 bits sufficient?

5. **Adaptation Strategy Selection**
   Problem: High-dimensional (1000 features), smooth landscape.
   Choose: Linear, Feedback, Diversity, or Self-Adaptive?
   Justify choice.

## Further Reading

**Adaptive GAs:**
1. **"Parameter Control in Evolutionary Algorithms" by Eiben et al.** - Comprehensive survey
2. **"Self-Adaptation in Evolutionary Algorithms" by Meyer-Nieberg & Beyer** - Theory
3. **"Adaptive and Self-Adaptive Evolutionary Algorithms"** - Practical guide

**Evolution Strategies:**
4. **"Evolution Strategies" by Beyer & Schwefel** - Self-adaptation origins
5. **"Covariance Matrix Adaptation Evolution Strategy (CMA-ES)"** - Advanced self-adaptation
6. **"Theory of Self-Adaptation"** - Mathematical foundations

**Feature Selection:**
7. **"Adaptive Feature Selection using GAs"** - Application domain
8. **"Dynamic Parameter Control in GAs"** - Implementation strategies
9. **"Online Parameter Adaptation"** - Real-time adjustment

**Advanced Topics:**
10. **"Hyper-Heuristics"** - Learning which operators to apply when
11. **"Meta-Evolution"** - Evolving evolutionary algorithms
12. **"Adaptive Operator Selection"** - Credit assignment for operators

---

*"Evolution adapts organisms. Adaptive GAs adapt themselves."*
---

**Next:** [Companion Slides](./03_adaptive_operators_slides.md) | [Notebook](../notebooks/02_hybrid_ga.ipynb)

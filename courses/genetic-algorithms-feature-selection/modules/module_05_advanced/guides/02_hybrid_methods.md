# Hybrid Methods: Memetic Algorithms and GA + Local Search

> **Reading time:** ~10 min | **Module:** 5 — Advanced Topics | **Prerequisites:** 01 Advanced Techniques

## In Brief

Hybrid methods combine genetic algorithms' global exploration with local search's exploitation, achieving faster convergence and better solutions than either method alone. Pure GAs explore broadly but converge slowly near optima. Pure local search (hill climbing, gradient descent) converges quickly but gets stuck in local optima. Memetic algorithms -- GAs with local search refinement -- inherit both strengths: global exploration from evolution, rapid local improvement from search.

<div class="callout-key">

**Key Concept Summary:** A memetic algorithm is a GA where each individual gets a chance to "study for the exam" (local search) before being graded (selection). The GA handles the global question -- "which region of the search space is promising?" -- while local search handles the local question -- "what is the best solution *near* this point?" The two mechanisms are complementary, not redundant. The result: 2-5x fewer evaluations to reach the same solution quality, or better final solutions within the same computational budget.

</div>

<div class="callout-insight">

GAs excel at finding "promising regions" of the search space but waste evaluations on marginal improvements. Local search excels at climbing to nearby peaks but has no escape mechanism. The synergy: GA identifies high-potential feature subsets (global), local search refines each to its local optimum (local), GA's crossover/selection moves between refined peaks. Result: 2-5x fewer evaluations to reach same quality, or better final solutions with same budget.

</div>

## When to Use Hybrid Methods

Hybrid methods add complexity over a pure GA, so they are not always the right choice. Use this decision framework to determine whether the added complexity is justified.

**USE hybrid methods when:**

**(a) Fitness evaluations are cheap relative to GA overhead.** Local search adds many additional fitness evaluations (one per feature per local search step). If your fitness function runs in milliseconds (e.g., a simple model on a small dataset), the extra evaluations are affordable and the local refinement pays off in solution quality. A GA with population 50 and local search refinement on the top 10 individuals, flipping each of 30 features once, adds 300 evaluations per generation -- trivial if each evaluation takes 10ms (3 seconds) but prohibitive if each takes 10 seconds (50 minutes).

**(b) The fitness landscape has many local optima that need refinement.** Feature selection landscapes are typically rugged: small changes to the feature set can cause significant fitness swings. A pure GA lands "near" good solutions but rarely hits the exact optimum because crossover and mutation make coarse moves. Local search performs the fine-grained exploration that pushes each candidate from "near the peak" to "on the peak."

**(c) You need high-precision solutions, not just good-enough.** If the difference between selecting features [1,3,5,7] and [1,3,5,8] matters for your application (e.g., regulatory requirements, model interpretability constraints), local search provides the refinement that a pure GA's stochastic exploration is unlikely to achieve within a reasonable generation budget.

**DO NOT use hybrid methods when:**

**(a) Fitness evaluations are expensive.** If training a model takes 5-10 seconds per individual, local search (which evaluates ~n_features neighbors per individual) becomes the dominant cost. A single local search step on 100 features costs 100 evaluations -- roughly two full generations of a 50-individual population. In this case, the computational budget is better spent on more GA generations or a larger population.

**(b) The fitness landscape is smooth.** If nearby feature subsets have similar fitness (e.g., because the model is robust to small feature changes), the GA converges to good solutions without local refinement. Local search adds cost but finds the same solutions the GA would eventually reach on its own.

<div class="callout-insight">

A practical test: run your pure GA and inspect the fitness improvement per generation in the final 10 generations. If fitness is still improving steadily, the GA has not converged and does not need local search. If fitness has plateaued but the average population fitness is far below the best, the GA is stuck near good solutions but not refining them -- this is the scenario where local search helps most.

</div>

![GA Lifecycle](./ga_lifecycle.svg)

## Formal Definition

### Memetic Algorithm Framework

**Standard GA:**
1. Initialize population
2. **Loop:** Evaluate fitness → Select → Crossover → Mutate → Replace
3. Return best individual

**Memetic Algorithm:**
1. Initialize population
2. **Loop:**
   - Evaluate fitness
   - **Local search refinement** ← Key addition
   - Select → Crossover → Mutate → Replace
3. Return best individual

### Local Search Operators

**Hill Climbing (1-bit flip neighborhood):**

Given chromosome $x$, neighbors are:
$$N(x) = \{x' : \text{HammingDistance}(x, x') = 1\}$$

For $n$-bit chromosome: $|N(x)| = n$ neighbors

**Search procedure:**
1. Start at current solution $x$
2. Evaluate all $n$ neighbors
3. Move to best neighbor if better than $x$
4. Repeat until no improvement

**k-opt Local Search:**

Neighborhood includes all solutions differing by $\leq k$ bits:
$$N_k(x) = \{x' : \text{HammingDistance}(x, x') \leq k\}$$

Size: $|N_k(x)| = \sum_{i=1}^k \binom{n}{i}$

For $k=2$, $n=100$: 5050 neighbors

### Lamarckian vs Baldwinian Learning

**Lamarckian:** Replace individual with locally improved version
$$x_{\text{new}} = \text{LocalSearch}(x)$$

Improved solution enters gene pool directly.

**Baldwinian:** Use improved fitness but keep original chromosome
$$\text{Fitness}(x) = \text{Fitness}(\text{LocalSearch}(x))$$
$$\text{Genes}(x) = x \text{ (unchanged)}$$

Fitness improves but genetic material unchanged.

### Computational Budget

Total evaluations:
$$E_{\text{total}} = E_{\text{GA}} + E_{\text{LS}}$$

Where:
- $E_{\text{GA}} = G \times P$ (generations × population)
- $E_{\text{LS}} = f_{\text{LS}} \times P \times G \times n_{\text{iter}}$

where $f_{\text{LS}}$ is fraction of individuals refined.

**Budget allocation:**
- All individuals: $f_{\text{LS}} = 1.0$ (expensive, thorough)
- Elite only: $f_{\text{LS}} = 0.1$ (cheap, focused)

## Intuitive Explanation

### The Complementary Strengths

**Pure GA:**
```
Generation 0:  [Random exploration across search space]
Generation 10: [Converging to regions...]
Generation 20: [Still converging...]
Generation 50: [Finally near optimum, but slow]
```

**Pure Hill Climbing:**
```
Start: Random point
Step 1: Found better neighbor → Move
Step 2: Found better neighbor → Move
Step 3: No better neighbors → STUCK (local optimum)
```

**Memetic Algorithm:**
```
Generation 0:  [Random exploration + local refinement]
                Each individual climbs to nearby peak
Generation 10: [GA explores between refined peaks]
                Crossover creates new starting points
                Local search refines each
Generation 20: [Converged to global optimum]
                Much faster than pure GA
```

### Feature Selection Example

**Pure GA (Generation 10):**
```
Individual 1: [1,0,1,1,0,1,0] → Fitness: 0.82
Individual 2: [1,0,1,1,0,0,1] → Fitness: 0.83
```

Small difference (one bit), but GA treats equally, slow convergence.

**Memetic Algorithm:**
```
Individual 1: [1,0,1,1,0,1,0] → Fitness: 0.82
  Local search:
    Try [0,0,1,1,0,1,0]: 0.80 ✗
    Try [1,1,1,1,0,1,0]: 0.84 ✓ (better!)
    Try [1,1,0,1,0,1,0]: 0.83 ✗
    ...
  Refined: [1,1,1,1,0,1,0] → Fitness: 0.84

Individual 2: [1,0,1,1,0,0,1] → Fitness: 0.83
  Local search → Refined: [1,0,1,1,1,0,1] → Fitness: 0.85

Next generation uses refined individuals → Faster convergence
```

### Lamarckian vs Baldwinian

**Scenario:** Individual with mediocre genes but lucky context.

**Lamarckian:**
```
Original genes: [1,0,0,1,0]  (mediocre)
After local search: [1,1,0,1,1]  (good)
→ Replace original with improved
→ [1,1,0,1,1] passes to offspring
```

Risk: Local improvements specific to current fitness landscape may not generalize.

**Baldwinian:**
```
Original genes: [1,0,0,1,0]  (mediocre)
After local search: [1,1,0,1,1]  (good)
→ Fitness calculated from improved
→ Original [1,0,0,1,0] passes to offspring
```

Benefit: Selection pressure toward "improvable" genotypes, not just locally optimal ones.

## Code Implementation

### Basic Memetic Algorithm


<span class="filename">__init__.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from typing import List, Tuple, Callable

class MemeticAlgorithm:
    """
    Memetic algorithm for feature selection.
    """

    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        population_size: int = 50,
        n_generations: int = 30,
        local_search_freq: float = 1.0,  # Fraction of population to refine
        local_search_iters: int = 10,
        learning_type: str = 'lamarckian',  # or 'baldwinian'
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.01,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.local_search_freq = local_search_freq
        self.local_search_iters = local_search_iters
        self.learning_type = learning_type
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = np.random.RandomState(random_state)

        # Tracking
        self.n_evaluations = 0
        self.n_local_improvements = 0

    def _hill_climbing(self, individual: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        1-flip hill climbing local search.

        Returns:
            (improved_individual, improved_fitness)
        """
        current = individual.copy()
        current_fitness = self.fitness_func(current)
        self.n_evaluations += 1

        for iteration in range(self.local_search_iters):
            improved = False

            # Try flipping each bit
            for i in range(self.n_features):
                neighbor = current.copy()
                neighbor[i] = 1 - neighbor[i]

                # Skip if no features selected
                if neighbor.sum() == 0:
                    continue

                neighbor_fitness = self.fitness_func(neighbor)
                self.n_evaluations += 1

                if neighbor_fitness > current_fitness:
                    current = neighbor
                    current_fitness = neighbor_fitness
                    improved = True
                    self.n_local_improvements += 1
                    break  # First improvement (faster)

            if not improved:
                break  # Local optimum reached

        return current, current_fitness

    def _local_search_refinement(
        self,
        population: List[np.ndarray],
        fitness_scores: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Apply local search to population.
        """
        # Determine which individuals to refine
        n_refine = max(1, int(self.population_size * self.local_search_freq))

        # Refine best individuals
        sorted_indices = np.argsort(fitness_scores)[-n_refine:]

        refined_population = population.copy()
        refined_fitness = fitness_scores.copy()

        for idx in sorted_indices:
            improved_ind, improved_fit = self._hill_climbing(population[idx])

            if self.learning_type == 'lamarckian':
                # Replace both genotype and fitness
                refined_population[idx] = improved_ind
                refined_fitness[idx] = improved_fit
            else:  # baldwinian
                # Update fitness only, keep original genotype
                refined_population[idx] = population[idx]
                refined_fitness[idx] = improved_fit

        return refined_population, refined_fitness

    def _tournament_selection(self, population: List[np.ndarray],
                              fitness_scores: np.ndarray) -> np.ndarray:
        """Select parent via tournament."""
        participants = self.rng.choice(self.population_size, 3, replace=False)
        best_idx = participants[fitness_scores[participants].argmax()]
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover."""
        if self.rng.random() > self.crossover_prob:
            return parent1.copy()

        mask = self.rng.random(self.n_features) < 0.5
        child = np.where(mask, parent1, parent2)

        if child.sum() == 0:
            child[self.rng.randint(self.n_features)] = 1

        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Bit-flip mutation."""
        mutant = individual.copy()
        for i in range(self.n_features):
            if self.rng.random() < self.mutation_prob:
                mutant[i] = 1 - mutant[i]

        if mutant.sum() == 0:
            mutant[self.rng.randint(self.n_features)] = 1

        return mutant

    def run(self) -> dict:
        """
        Run memetic algorithm.
        """
        # Initialize population
        population = [
            (self.rng.random(self.n_features) < 0.3).astype(int)
            for _ in range(self.population_size)
        ]

        for ind in population:
            if ind.sum() == 0:
                ind[self.rng.randint(self.n_features)] = 1

        # Evaluate initial population
        fitness_scores = np.array([self.fitness_func(ind) for ind in population])
        self.n_evaluations += self.population_size

        best_fitness_history = []
        avg_fitness_history = []

        # Evolution
        for generation in range(self.n_generations):
            # Local search refinement
            population, fitness_scores = self._local_search_refinement(
                population, fitness_scores
            )

            # Track progress
            best_fitness = fitness_scores.max()
            avg_fitness = fitness_scores.mean()
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            if generation % 10 == 0:
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Evals={self.n_evaluations}")

            # Generate offspring
            offspring = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)

            # Evaluate offspring
            offspring_fitness = np.array([self.fitness_func(ind) for ind in offspring])
            self.n_evaluations += self.population_size

            # Elitism: keep best individual
            best_idx = fitness_scores.argmax()
            worst_offspring_idx = offspring_fitness.argmin()
            offspring[worst_offspring_idx] = population[best_idx]
            offspring_fitness[worst_offspring_idx] = fitness_scores[best_idx]

            # Replace population
            population = offspring
            fitness_scores = offspring_fitness

        best_idx = fitness_scores.argmax()

        return {
            'best_individual': population[best_idx],
            'best_fitness': fitness_scores[best_idx],
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'n_evaluations': self.n_evaluations,
            'n_local_improvements': self.n_local_improvements
        }


# Example usage
print("=" * 70)
print("MEMETIC ALGORITHM VS STANDARD GA")
print("=" * 70)

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate data
X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=10,
    n_redundant=15,
    random_state=42
)

def fitness_function(chromosome):
    """Feature selection fitness."""
    if chromosome.sum() == 0:
        return 0

    selected = chromosome.astype(bool)
    X_selected = X[:, selected]

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    score = cross_val_score(clf, X_selected, y, cv=3).mean()

    # Penalty for too many features
    penalty = 0.001 * chromosome.sum()

    return score - penalty

# Standard GA (baseline)
from genetic_algorithms_feature_selection.modules.module_01_ga_fundamentals.guides.ga_components import GeneticAlgorithm

print("\n" + "-" * 70)
print("Standard GA:")
print("-" * 70)

ga_standard = GeneticAlgorithm(
    n_features=30,
    fitness_func=fitness_function,
    population_size=50,
    n_generations=20,
    random_state=42
)
result_ga = ga_standard.run()

print(f"\nResults:")
print(f"  Best fitness: {result_ga['best_fitness']:.4f}")
print(f"  Features selected: {result_ga['best_individual'].sum()}")

# Memetic Algorithm (Lamarckian)
print("\n" + "-" * 70)
print("Memetic Algorithm (Lamarckian):")
print("-" * 70)

ma_lamarckian = MemeticAlgorithm(
    n_features=30,
    fitness_func=fitness_function,
    population_size=50,
    n_generations=20,
    local_search_freq=0.2,  # Refine top 20%
    local_search_iters=5,
    learning_type='lamarckian',
    random_state=42
)
result_ma_lamarck = ma_lamarckian.run()

print(f"\nResults:")
print(f"  Best fitness: {result_ma_lamarck['best_fitness']:.4f}")
print(f"  Features selected: {result_ma_lamarck['best_individual'].sum()}")
print(f"  Total evaluations: {result_ma_lamarck['n_evaluations']}")
print(f"  Local improvements: {result_ma_lamarck['n_local_improvements']}")

# Memetic Algorithm (Baldwinian)
print("\n" + "-" * 70)
print("Memetic Algorithm (Baldwinian):")
print("-" * 70)

ma_baldwinian = MemeticAlgorithm(
    n_features=30,
    fitness_func=fitness_function,
    population_size=50,
    n_generations=20,
    local_search_freq=0.2,
    local_search_iters=5,
    learning_type='baldwinian',
    random_state=42
)
result_ma_baldwin = ma_baldwinian.run()

print(f"\nResults:")
print(f"  Best fitness: {result_ma_baldwin['best_fitness']:.4f}")
print(f"  Features selected: {result_ma_baldwin['best_individual'].sum()}")
print(f"  Total evaluations: {result_ma_baldwin['n_evaluations']}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Standard GA:            {result_ga['best_fitness']:.4f}")
print(f"Memetic (Lamarckian):   {result_ma_lamarck['best_fitness']:.4f}")
print(f"Memetic (Baldwinian):   {result_ma_baldwin['best_fitness']:.4f}")
```

</div>
</div>


### Advanced Hybrid: GA + Greedy Forward Selection

```python
class GAGreedyHybrid:
    """
    Hybrid of GA and greedy forward selection.
    """

    def greedy_forward_selection(
        self,
        individual: np.ndarray,
        max_features: int = 5
    ) -> np.ndarray:
        """
        Greedy forward selection starting from individual.

        Add features one at a time that maximize improvement.
        """
        selected = individual.copy()
        current_fitness = self.fitness_func(selected)

        for _ in range(max_features):
            best_improvement = 0
            best_feature = None

            # Try adding each unselected feature
            unselected = np.where(selected == 0)[0]

            for feature in unselected:
                candidate = selected.copy()
                candidate[feature] = 1

                candidate_fitness = self.fitness_func(candidate)

                if candidate_fitness > current_fitness + best_improvement:
                    best_improvement = candidate_fitness - current_fitness
                    best_feature = feature

            if best_feature is not None:
                selected[best_feature] = 1
                current_fitness += best_improvement
            else:
                break  # No improvement found

        return selected


# Example integration
class HybridGA(MemeticAlgorithm, GAGreedyHybrid):
    """Combine memetic with greedy search."""
    pass
```

<div class="callout-danger">

<strong>Danger:</strong> Applying local search to every individual in every generation makes the GA prohibitively expensive (n_features evaluations per local search step). Limit local search to the top 10-20% of the population, or apply it every 5-10 generations.

</div>

## Common Pitfalls

<div class="callout-warning">

⚠️ **Warning:** Memetic algorithms can silently consume 5-10x more fitness evaluations than a pure GA. Always compare methods at equal evaluation budgets, not equal generations, to get an honest performance comparison.

</div>

**1. Excessive Local Search**
- Problem: Refining every individual every generation
- Symptom: 10× more evaluations, only marginal improvement
- Solution: Refine elite only (top 10-20%), or every N generations

**2. Local Search Cancels Crossover**
- Problem: Local search immediately erases genetic diversity from crossover
- Symptom: Population converges prematurely
- Solution: Apply local search selectively, maintain diversity pressure

**3. Wrong Learning Type**
- Problem: Using Lamarckian when problem has fitness landscape drift
- Symptom: Good solutions early become bad later, population stuck
- Solution: Use Baldwinian for non-stationary problems

**4. Infinite Local Search**
- Problem: Hill climbing with no iteration limit
- Symptom: Single individual takes hours to refine
- Solution: Hard limit on local search iterations (5-20)

**5. Ignoring Computational Budget**
- Problem: Not accounting for local search evaluations in total budget
- Symptom: "Memetic uses same generations" but actually 5× more evaluations
- Solution: Track total evaluations, compare at equal budgets

## Connections

<div class="callout-info">

ℹ️ **How this connects to the rest of the course:**

</div>

**Builds on:**
- Module 1-3: Core GA operators (what to hybridize)
- Module 2: Fitness functions (local search uses same fitness)
- Optimization theory (local vs global search)

**Leads to:**
- Module 5.3: Adaptive operators (learning when to apply local search)
- Multi-objective optimization (Pareto local search)
- Ensemble methods (combine multiple local search strategies)

**Related concepts:**
- Simulated annealing (alternative local search)
- Tabu search (memory-based local search)
- Variable neighborhood search (multiple neighborhood structures)

<div class="callout-key">

<strong>Key Takeaway:</strong> Memetic algorithms (GA + local search) are the practical gold standard for feature selection. The GA finds promising regions, local search polishes each candidate to its local optimum. This combination typically finds solutions in 2-5x fewer evaluations than either method alone.

</div>

## Practice Problems

1. **Budget Allocation**
   Total budget: 5000 evaluations
   Population: 50, Generations: 50 (2500 GA evaluations)
   Local search: How many iterations per individual to stay in budget?

2. **Lamarckian vs Baldwinian**
   Problem: Feature selection for time series (non-stationary)
   Which learning type? Why?
   What happens if wrong choice?

3. **Neighborhood Size**
   100 features, 1-flip neighborhood: 100 neighbors
   100 features, 2-flip neighborhood: 5050 neighbors
   Which is better? Trade-offs?

4. **Hybrid Design**
   You have access to: (a) GA, (b) Hill climbing, (c) Simulated annealing
   Design a 3-stage hybrid. What order? Why?

5. **Local Search Frequency**
   Refine all individuals: High quality, slow
   Refine top 10%: Fast, misses some improvements
   Refine every 5 generations: Batch efficiency
   Which strategy for 1-hour time limit?

6. **Conceptual: When NOT to Hybridize**
   You have a GA with fitness function that trains a deep neural network (30 seconds per evaluation), population size 50, and 30 generations. A colleague suggests adding Lamarckian local search to the top 20% of the population. Calculate the additional evaluations per generation and total added time. Is this a good idea? What alternative would you suggest?

7. **Conceptual: Lamarckian vs Baldwinian**
   In a non-stationary feature selection problem (the optimal features change as new data arrives monthly), which learning type -- Lamarckian or Baldwinian -- is more robust? Explain what goes wrong with the other choice.

## Further Reading

**Memetic Algorithms:**
1. **"Memetic Algorithms: A Short Introduction" by Moscato & Cotta** - Foundational paper
2. **"Handbook of Memetic Algorithms"** - Comprehensive reference
3. **"Memetic Algorithms for Feature Selection"** - Application to FS

**Hybrid Methods:**
4. **"Hybrid Metaheuristics" by Blum et al.** - Combining metaheuristics
5. **"Local Search in Evolutionary Algorithms"** - Integration strategies
6. **"Lamarckian and Baldwinian Evolution"** - Learning types

**Feature Selection:**
7. **"Wrapper Feature Selection with Memetic Algorithms"** - Practical applications
8. **"Greedy Forward Selection Meets Genetic Algorithms"** - Hybrid designs
9. **"Multi-Stage Hybrid Feature Selection"** - Complex hybrids

**Advanced Topics:**
10. **"Adaptive Memetic Algorithms"** - Learning when to apply local search
11. **"Parallel Memetic Algorithms"** - Distributed implementations
12. **"Memetic Algorithms for Multi-Objective Optimization"** - Extensions

*"Evolution finds the mountain. Local search climbs it."*

---

**Next:** [Companion Slides](./02_hybrid_methods_slides.md) | [Notebook](../notebooks/02_hybrid_ga.ipynb)

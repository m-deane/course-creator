# Production Considerations: Scaling, Parallelization, and Reproducibility

> **Reading time:** ~10 min | **Module:** 4 — Implementation | **Prerequisites:** 02 Custom Operators

## In Brief

Production genetic algorithms require careful attention to computational efficiency, reproducibility, and system integration. Naive GA implementations are prohibitively slow for high-dimensional feature spaces (hours for 1000 features), non-reproducible due to random seeds, and difficult to integrate with ML pipelines. Production-ready GAs use parallel fitness evaluation, efficient caching, deterministic execution, and standard ML interfaces.

<div class="callout-insight">
The fitness function is the bottleneck—evaluating a single individual requires training a full ML model. With population size 100 and 50 generations, that's 5000 model training runs. Parallelization across individuals reduces wall-clock time linearly with available cores. Combined with fitness caching (avoiding duplicate evaluations), warm-start models, and early stopping, production GAs achieve 10-100× speedup while maintaining determinism through careful random state management.
</div>


![GA Lifecycle](./ga_lifecycle.svg)

## Formal Definition

### Computational Complexity

**Serial GA:**
$$T_{\text{serial}} = G \times P \times T_{\text{fitness}}$$

Where:
- $G$: Number of generations
- $P$: Population size
- $T_{\text{fitness}}$: Time per fitness evaluation

**Parallel GA (Master-Worker):**
$$T_{\text{parallel}} = G \times \frac{P}{W} \times T_{\text{fitness}} + G \times T_{\text{comm}}$$

Where:
- $W$: Number of workers
- $T_{\text{comm}}$: Communication overhead

**Speedup:**
$$S = \frac{T_{\text{serial}}}{T_{\text{parallel}}} \approx \frac{W}{1 + \frac{W \times T_{\text{comm}}}{P \times T_{\text{fitness}}}}$$

Ideal: $S = W$ (linear speedup)
Reality: $S < W$ (overhead, load imbalance)

### Caching Efficiency

**Cache Hit Rate:**
$$r_{\text{cache}} = \frac{N_{\text{duplicate}}}{N_{\text{total}}}$$

For similar individuals (Hamming distance = 1):
- No cache: Evaluate both → $2 \times T_{\text{fitness}}$
- With cache: Evaluate once → $T_{\text{fitness}}$

**Expected Cache Savings:**
$$T_{\text{saved}} = r_{\text{cache}} \times N_{\text{total}} \times T_{\text{fitness}}$$

### Reproducibility Requirements

**Deterministic Execution:**
1. Fixed random seeds at initialization
2. Consistent operator ordering
3. Reproducible fitness function
4. Version-controlled data and code

**Random State Management:**
$$\text{RNG}_{\text{child}} = \text{seed}(\text{RNG}_{\text{parent}}, \text{context})$$

## Intuitive Explanation

### The Parallelization Problem

**Serial Execution:**
```
Generation 1:
  Evaluate ind 1: 10s
  Evaluate ind 2: 10s
  ...
  Evaluate ind 100: 10s
Total: 1000s (16 minutes)

50 generations → 50 × 16 = 800 minutes (13 hours)
```

**Parallel Execution (8 cores):**
```
Generation 1:
  Batch 1 (inds 1-8):   10s (parallel)
  Batch 2 (inds 9-16):  10s (parallel)
  ...
  Batch 13 (inds 97-100): 10s (4 cores idle)
Total: 130s (2 minutes)

50 generations → 50 × 2 = 100 minutes (1.7 hours)

Speedup: 13 / 1.7 ≈ 7.6× (close to 8× ideal)
```

### Fitness Caching Benefits

**Without Cache:**
```
Generation 10:
  Individual A: [1,0,1,0,1] → Train model → Fitness: 0.85
  Individual B: [1,0,1,0,1] → Train model → Fitness: 0.85 (duplicate!)
  2 model training runs
```

**With Cache:**
```
Generation 10:
  Individual A: [1,0,1,0,1] → Train model → Fitness: 0.85 → Cache
  Individual B: [1,0,1,0,1] → Cache hit! → Fitness: 0.85
  1 model training run (50% savings)
```

Over 5000 evaluations with 20% cache hit rate → 1000 evaluations saved!

### Reproducibility Challenge

**Non-Reproducible Run:**
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Run 1
np.random.seed(42)
ga.run()  # Best fitness: 0.87

# Run 2
np.random.seed(42)
ga.run()  # Best fitness: 0.89  ← Different!
```
</div>


Why? Cross-validation shuffles differently, thread scheduling varies, floating-point operations non-deterministic.

**Reproducible Run:**
```python
# Fix ALL randomness sources
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'
tf.random.set_seed(42)  # If using TensorFlow

ga.run()  # Always same result
```

## Code Implementation

### Parallel Fitness Evaluation

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import multiprocessing as mp
from typing import List, Callable
import time
import hashlib
import pickle

class ParallelGA:
    """
    Genetic algorithm with parallel fitness evaluation.
    """

    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        population_size: int = 100,
        n_generations: int = 50,
        n_jobs: int = -1,  # -1 = use all cores
        cache_fitness: bool = True,
        random_state: int = 42
    ):
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.cache_fitness = cache_fitness
        self.random_state = random_state

        # Initialize random state
        self.rng = np.random.RandomState(random_state)

        # Fitness cache
        self.fitness_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance tracking
        self.evaluation_times = []

    def _chromosome_hash(self, chromosome: np.ndarray) -> str:
        """Generate hash for caching."""
        return hashlib.sha256(chromosome.tobytes()).hexdigest()

    def _evaluate_fitness(self, chromosome: np.ndarray) -> float:
        """
        Evaluate fitness with caching.
        """
        if self.cache_fitness:
            chrom_hash = self._chromosome_hash(chromosome)

            if chrom_hash in self.fitness_cache:
                self.cache_hits += 1
                return self.fitness_cache[chrom_hash]

            self.cache_misses += 1

        # Evaluate fitness
        start_time = time.time()
        fitness = self.fitness_func(chromosome)
        eval_time = time.time() - start_time
        self.evaluation_times.append(eval_time)

        if self.cache_fitness:
            self.fitness_cache[chrom_hash] = fitness

        return fitness

    def evaluate_population(self, population: List[np.ndarray]) -> np.ndarray:
        """
        Evaluate entire population in parallel.
        """
        if self.n_jobs == 1:
            # Serial execution
            fitness_scores = [self._evaluate_fitness(ind) for ind in population]
        else:
            # Parallel execution
            fitness_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_fitness)(ind) for ind in population
            )

        return np.array(fitness_scores)

    def run(self) -> dict:
        """
        Run GA with parallel fitness evaluation.
        """
        start_time = time.time()

        # Initialize population
        population = [
            (self.rng.random(self.n_features) < 0.3).astype(int)
            for _ in range(self.population_size)
        ]

        # Ensure at least one feature per individual
        for ind in population:
            if ind.sum() == 0:
                ind[self.rng.randint(self.n_features)] = 1

        best_fitness_history = []
        avg_fitness_history = []

        # Evolution
        for generation in range(self.n_generations):
            gen_start = time.time()

            # Evaluate fitness (parallel)
            fitness_scores = self.evaluate_population(population)

            # Track progress
            best_fitness = fitness_scores.max()
            avg_fitness = fitness_scores.mean()
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            gen_time = time.time() - gen_start

            if generation % 10 == 0:
                cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if self.cache_misses > 0 else 0
                print(f"Gen {generation:3d}: Best={best_fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}, Time={gen_time:.1f}s, "
                      f"Cache={cache_rate*100:.1f}%")

            # Selection, crossover, mutation (simplified)
            # [Would implement full GA operators here]

        total_time = time.time() - start_time

        return {
            'best_fitness': max(best_fitness_history),
            'best_individual': population[fitness_scores.argmax()],
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'total_time': total_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if self.cache_misses > 0 else 0,
            'avg_eval_time': np.mean(self.evaluation_times),
            'total_evaluations': len(self.evaluation_times)
        }


# Example: Compare serial vs parallel
print("=" * 70)
print("PARALLEL GA COMPARISON")
print("=" * 70)

# Dummy fitness function (simulates ML model training)
def fitness_function(chromosome):
    """Simulated expensive fitness evaluation."""
    if chromosome.sum() == 0:
        return 0

    # Simulate computation
    time.sleep(0.1)  # 100ms per evaluation

    # Return random fitness
    return np.random.random()

n_features = 50
population_size = 50
n_generations = 5

# Serial execution
print("\nSerial Execution:")
ga_serial = ParallelGA(
    n_features=n_features,
    fitness_func=fitness_function,
    population_size=population_size,
    n_generations=n_generations,
    n_jobs=1,
    cache_fitness=False,
    random_state=42
)
result_serial = ga_serial.run()

print(f"\nResults:")
print(f"  Total time: {result_serial['total_time']:.1f}s")
print(f"  Evaluations: {result_serial['total_evaluations']}")
print(f"  Avg eval time: {result_serial['avg_eval_time']*1000:.0f}ms")

# Parallel execution
print("\n" + "-" * 70)
print("Parallel Execution (all cores):")
ga_parallel = ParallelGA(
    n_features=n_features,
    fitness_func=fitness_function,
    population_size=population_size,
    n_generations=n_generations,
    n_jobs=-1,
    cache_fitness=False,
    random_state=42
)
result_parallel = ga_parallel.run()

print(f"\nResults:")
print(f"  Total time: {result_parallel['total_time']:.1f}s")
print(f"  Evaluations: {result_parallel['total_evaluations']}")
print(f"  Speedup: {result_serial['total_time'] / result_parallel['total_time']:.1f}×")

# Parallel with caching
print("\n" + "-" * 70)
print("Parallel + Caching:")
ga_cached = ParallelGA(
    n_features=n_features,
    fitness_func=fitness_function,
    population_size=population_size,
    n_generations=n_generations,
    n_jobs=-1,
    cache_fitness=True,
    random_state=42
)
result_cached = ga_cached.run()

print(f"\nResults:")
print(f"  Total time: {result_cached['total_time']:.1f}s")
print(f"  Cache hit rate: {result_cached['cache_hit_rate']*100:.1f}%")
print(f"  Speedup vs serial: {result_serial['total_time'] / result_cached['total_time']:.1f}×")
```

### Reproducibility Management

```python
import random
import os

class ReproducibleGA:
    """
    GA with complete reproducibility.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._setup_reproducibility()

    def _setup_reproducibility(self):
        """
        Configure all randomness sources.
        """
        # Python random
        random.seed(self.random_state)

        # NumPy
        np.random.seed(self.random_state)

        # Hash seed (for dictionary ordering)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

        # If using other libraries
        try:
            import tensorflow as tf
            tf.random.set_seed(self.random_state)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(self.random_state)
        except ImportError:
            pass

    def save_state(self, filepath: str):
        """Save GA state for reproducibility."""
        state = {
            'random_state': self.random_state,
            'numpy_state': np.random.get_state(),
            'python_state': random.getstate()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """Load GA state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.random_state = state['random_state']
        np.random.set_state(state['numpy_state'])
        random.setstate(state['python_state'])


# Test reproducibility
print("\n" + "=" * 70)
print("REPRODUCIBILITY TEST")
print("=" * 70)

def run_ga_sample(seed):
    """Run simple GA and return best fitness."""
    ga = ReproducibleGA(random_state=seed)
    population = [np.random.randint(0, 2, 10) for _ in range(20)]
    fitness = [np.random.random() for _ in range(20)]
    return max(fitness), population[0]

# Run twice with same seed
result1_fitness, result1_pop = run_ga_sample(42)
result2_fitness, result2_pop = run_ga_sample(42)

print(f"\nRun 1 best fitness: {result1_fitness:.6f}")
print(f"Run 2 best fitness: {result2_fitness:.6f}")
print(f"Identical: {result1_fitness == result2_fitness}")
print(f"\nRun 1 first ind: {result1_pop}")
print(f"Run 2 first ind: {result2_pop}")
print(f"Identical: {np.array_equal(result1_pop, result2_pop)}")
```

### Production Integration

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class GAFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible GA feature selector.
    """

    def __init__(
        self,
        estimator,
        population_size: int = 50,
        n_generations: int = 30,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.01,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        self.estimator = estimator
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit GA feature selector.
        """
        self.n_features_in_ = X.shape[1]

        # Define fitness function
        def fitness(chromosome):
            if chromosome.sum() == 0:
                return 0

            selected = chromosome.astype(bool)
            X_selected = X[:, selected]

            # Cross-validation score
            score = cross_val_score(
                self.estimator,
                X_selected,
                y,
                cv=3,
                n_jobs=1  # Nested parallelism handled by GA
            ).mean()

            return score

        # Run GA
        ga = ParallelGA(
            n_features=self.n_features_in_,
            fitness_func=fitness,
            population_size=self.population_size,
            n_generations=self.n_generations,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

        result = ga.run()

        # Store results
        self.best_individual_ = result['best_individual']
        self.best_fitness_ = result['best_fitness']
        self.support_ = self.best_individual_.astype(bool)
        self.n_features_ = self.support_.sum()

        return self

    def transform(self, X):
        """Select features."""
        return X[:, self.support_]

    def fit_transform(self, X, y):
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        """Get selected feature indices or mask."""
        if indices:
            return np.where(self.support_)[0]
        return self.support_


# Example: Integration with sklearn pipeline
print("\n" + "=" * 70)
print("SKLEARN PIPELINE INTEGRATION")
print("=" * 70)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(
    n_samples=200,
    n_features=20,
    n_informative=5,
    n_redundant=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nDataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# Build pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ga_selector', GAFeatureSelector(
        estimator=RandomForestClassifier(random_state=42),
        population_size=20,
        n_generations=5,
        n_jobs=4,
        random_state=42
    )),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Fit pipeline
print("\nFitting pipeline...")
pipeline.fit(X_train, y_train)

# Results
selected_features = pipeline.named_steps['ga_selector'].get_support(indices=True)
n_selected = len(selected_features)

print(f"\nSelected {n_selected} features: {selected_features}")
print(f"Best fitness: {pipeline.named_steps['ga_selector'].best_fitness_:.4f}")

# Test performance
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"\nPerformance:")
print(f"  Train accuracy: {train_score:.4f}")
print(f"  Test accuracy: {test_score:.4f}")
```

## Common Pitfalls

<div class="callout-warning">

⚠️ **Warning:** Parallel fitness evaluation with shared mutable state (e.g., a global cache dictionary) causes race conditions and non-deterministic results. Use process-safe data structures or evaluate cache checks before dispatching parallel work.

</div>

**1. Race Conditions in Parallel Execution**
- Problem: Multiple workers modifying shared state simultaneously
- Symptom: Inconsistent results, crashes, deadlocks
- Solution: Use immutable data, message passing, proper synchronization

**2. Overhead Dominates with Small Fitness Functions**
- Problem: Fitness evaluation is 10ms, but communication overhead is 50ms
- Symptom: Parallel slower than serial
- Solution: Batch evaluations, use threads instead of processes, increase population size

**3. Non-Deterministic Fitness Functions**
- Problem: Cross-validation uses different folds each time
- Symptom: Same chromosome gets different fitness
- Solution: Fix CV splitter random state, use stratified splits

**4. Memory Leaks in Long Runs**
- Problem: Cache grows unbounded over generations
- Symptom: Out of memory after 1000 generations
- Solution: LRU cache, periodic cache clearing, memory profiling

**5. Incorrect Speedup Expectations**
- Problem: Expecting 8× speedup with 8 cores
- Symptom: Actual speedup is 5×, perceived as failure
- Solution: Understand Amdahl's law, account for overhead, measure realistic baselines

## Connections

<div class="callout-info">
ℹ️ **How this connects to the rest of the course:**
</div>

**Builds on:**
- Module 1-3: Core GA implementation (what to parallelize)
- Module 4.1: Sklearn integration (production interfaces)
- System architecture (parallel computing, caching)

**Leads to:**
- Module 5: Advanced methods (distributed GAs, island models)
- ML ops (model deployment, monitoring)
- Cloud deployment (scalable GA on clusters)

**Related concepts:**
- Distributed computing (Dask, Ray, Spark for large-scale GA)
- Hyperparameter optimization (Optuna, Hyperopt parallelization)
- Workflow orchestration (Airflow, Prefect for GA pipelines)

## Practice Problems

1. **Speedup Calculation**
   Serial GA: 2 hours
   Parallel GA (8 cores): 20 minutes
   What is actual speedup? Why not 8×?
   What percent is communication overhead?

2. **Cache Design**
   100 features, population 100, 50 generations.
   Estimate: How many unique individuals?
   Expected cache hit rate after generation 10?
   Memory required for cache?

3. **Reproducibility Bug**
   Two runs with same seed produce different results.
   Fitness function uses cross-validation with cv=5.
   What's the likely cause?
   How to fix?

4. **Load Balancing**
   8 workers, population 50.
   Some individuals evaluate in 5s, others in 15s.
   Best assignment strategy?
   Expected idle time?

5. **Production Deployment**
   Deploy GA for daily feature selection on 10000 features.
   Requirements: <1 hour runtime, reproducible, fault-tolerant.
   Architecture design? Key components?

## Further Reading

**Parallel Computing:**
1. **"Parallel Genetic Algorithms: Theory and Applications"** by Cantú-Paz - Comprehensive parallel GA theory
2. **"An Introduction to Parallel Programming" by Pacheco** - Parallel computing fundamentals
3. **"Joblib Documentation"** - Python parallelization library

**Reproducibility:**
4. **"Reproducible Research in Computational Science"** - Best practices
5. **"Random Seeds in Machine Learning"** - Ensuring reproducibility
6. **"MLflow Documentation"** - Experiment tracking and reproducibility

**Production ML:**
7. **"Building Machine Learning Pipelines" by Hapke & Nelson** - Production ML systems
8. **"Designing Data-Intensive Applications" by Kleppmann** - Distributed systems
9. **"MLOps: Continuous Delivery for ML" by Treveil et al.** - Production ML lifecycle

**Performance Optimization:**
10. **"High Performance Python" by Gorelick & Ozsvald** - Python performance tuning
11. **"Dask Documentation"** - Distributed computing in Python
12. **"Ray Documentation"** - Scalable distributed applications

---

*"Production code is 10% algorithm, 90% engineering."*
---

**Next:** [Companion Slides](./03_production_considerations_slides.md) | [Notebook](../notebooks/03_case_study.ipynb)

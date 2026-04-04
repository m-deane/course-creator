# Cross-Validation Based Fitness Functions

> **Reading time:** ~7 min | **Module:** 2 — Fitness Functions | **Prerequisites:** 01 Fitness Functions

## Introduction

For feature selection, fitness must evaluate how well selected features predict the target. Cross-validation provides robust estimation of generalization performance.


![Fitness Landscape](./fitness_landscape.svg)

## The CV-Based Fitness Framework

### Basic Structure

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">__init__.py</span>
</div>

```python
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

class CVFitnessEvaluator:
    """
    Evaluate feature subsets using cross-validation.
    """

    def __init__(self, X, y, model, cv=5, metric='neg_mse',
                 penalty_weight=0.01, min_features=1):
        """
        Parameters:
        -----------
        X : array-like
            Full feature matrix
        y : array-like
            Target variable
        model : sklearn estimator
            Model to evaluate features
        cv : int or CV splitter
            Cross-validation strategy
        metric : str
            Scoring metric
        penalty_weight : float
            Weight for feature count penalty
        min_features : int
            Minimum features required
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.model = model
        self.cv = cv
        self.metric = metric
        self.penalty_weight = penalty_weight
        self.min_features = min_features

        self.n_features = X.shape[1]
        self.evaluation_count = 0
        self.cache = {}

    def evaluate(self, chromosome):
        """
        Evaluate fitness of a feature subset.

        Parameters:
        -----------
        chromosome : list
            Binary vector indicating selected features

        Returns:
        --------
        fitness : float
            Higher is better
        """
        self.evaluation_count += 1

        # Check cache
        key = tuple(chromosome)
        if key in self.cache:
            return self.cache[key]

        # Get selected features
        selected = [i for i, x in enumerate(chromosome) if x == 1]

        # Handle edge cases
        if len(selected) < self.min_features:
            return -np.inf

        # Subset data
        X_subset = self.X[:, selected]

        try:
            # Cross-validation score
            scores = cross_val_score(
                self.model, X_subset, self.y,
                cv=self.cv, scoring=self.metric
            )
            cv_score = scores.mean()

            # Apply penalty for more features
            n_selected = len(selected)
            penalty = self.penalty_weight * n_selected / self.n_features

            fitness = cv_score - penalty

        except Exception as e:
            fitness = -np.inf

        # Cache result
        self.cache[key] = fitness

        return fitness

    def get_stats(self):
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'unique_evaluations': len(self.cache),
            'cache_hit_rate': 1 - len(self.cache) / max(self.evaluation_count, 1)
        }


# Example usage
from sklearn.datasets import make_regression

# Generate synthetic data
np.random.seed(42)
X, y = make_regression(
    n_samples=500, n_features=50, n_informative=10,
    noise=10, random_state=42
)

# Create evaluator
evaluator = CVFitnessEvaluator(
    X, y,
    model=Ridge(alpha=1.0),
    cv=5,
    metric='neg_mean_squared_error',
    penalty_weight=0.01
)

# Test with random chromosomes
test_chromosomes = [
    [1] * 10 + [0] * 40,  # First 10 features
    [1] * 50,              # All features
    [1 if i % 5 == 0 else 0 for i in range(50)],  # Every 5th feature
]

for chrom in test_chromosomes:
    fitness = evaluator.evaluate(chrom)
    n_features = sum(chrom)
    print(f"Features: {n_features:2d}, Fitness: {fitness:.4f}")
```
</div>


![Walk-Forward Timeline](./walk_forward_timeline.svg)

<div class="callout-warning">

⚠️ **Warning:** Caching fitness values assumes deterministic evaluation. If your cross-validation uses random splits or your model has stochastic initialization, two evaluations of the same chromosome can return different fitness values, breaking the cache invariant.

</div>

## Time Series Cross-Validation

For financial applications, use time-aware CV:

```python
class TimeSeriesFitnessEvaluator(CVFitnessEvaluator):
    """
    Fitness evaluation with time series cross-validation.
    """

    def __init__(self, X, y, model, n_splits=5, test_size=None,
                 gap=0, metric='neg_mse', **kwargs):
        """
        Parameters:
        -----------
        n_splits : int
            Number of train/test splits
        test_size : int
            Size of test set in each fold
        gap : int
            Gap between train and test (prevent lookahead)
        """
        # Create time series CV splitter
        cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )

        super().__init__(X, y, model, cv=cv, metric=metric, **kwargs)

    def evaluate_with_details(self, chromosome):
        """
        Return detailed evaluation results.
        """
        selected = [i for i, x in enumerate(chromosome) if x == 1]

        if len(selected) < self.min_features:
            return {
                'fitness': -np.inf,
                'cv_scores': [],
                'selected_features': selected
            }

        X_subset = self.X[:, selected]

        scores = cross_val_score(
            self.model, X_subset, self.y,
            cv=self.cv, scoring=self.metric
        )

        fitness = scores.mean() - self.penalty_weight * len(selected) / self.n_features

        return {
            'fitness': fitness,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores.tolist(),
            'n_features': len(selected),
            'selected_features': selected
        }


# Example with time series data
# Simulate time series features and target
n_samples = 500
X_ts = np.random.randn(n_samples, 30)
y_ts = (
    0.5 * X_ts[:, 0] +
    0.3 * X_ts[:, 1] +
    0.2 * X_ts[:, 2] +
    np.random.randn(n_samples) * 0.5
)

ts_evaluator = TimeSeriesFitnessEvaluator(
    X_ts, y_ts,
    model=Ridge(alpha=1.0),
    n_splits=5,
    test_size=50,
    gap=5,  # 5-day gap between train and test
    penalty_weight=0.02
)

# Evaluate
chrom = [1, 1, 1] + [0] * 27  # First 3 features
details = ts_evaluator.evaluate_with_details(chrom)

print("Time Series CV Evaluation:")
print(f"  Fitness: {details['fitness']:.4f}")
print(f"  CV Mean: {details['cv_mean']:.4f}")
print(f"  CV Std: {details['cv_std']:.4f}")
print(f"  Scores: {[f'{s:.4f}' for s in details['cv_scores']]}")
```

## Multi-Objective Fitness

Balance prediction accuracy vs. feature parsimony:

```python
class MultiObjectiveFitness:
    """
    Multi-objective fitness for feature selection.

    Objectives:
    1. Maximize prediction accuracy (CV score)
    2. Minimize number of features
    """

    def __init__(self, X, y, model, cv=5, metric='neg_mse'):
        self.X = np.array(X)
        self.y = np.array(y)
        self.model = model
        self.cv = cv
        self.metric = metric
        self.n_features = X.shape[1]

    def evaluate(self, chromosome):
        """
        Return both objectives.

        Returns:
        --------
        (accuracy, -n_features) : tuple
            Both objectives to maximize (hence negative features)
        """
        selected = [i for i, x in enumerate(chromosome) if x == 1]

        if len(selected) == 0:
            return (-np.inf, 0)

        X_subset = self.X[:, selected]

        try:
            scores = cross_val_score(
                self.model, X_subset, self.y,
                cv=self.cv, scoring=self.metric
            )
            accuracy = scores.mean()
        except:
            accuracy = -np.inf

        # Both objectives to maximize
        return (accuracy, -len(selected))

    def dominates(self, obj1, obj2):
        """Check if obj1 Pareto-dominates obj2."""
        return all(o1 >= o2 for o1, o2 in zip(obj1, obj2)) and \
               any(o1 > o2 for o1, o2 in zip(obj1, obj2))

    def get_pareto_front(self, population):
        """Get Pareto-optimal solutions."""
        objectives = [self.evaluate(ind) for ind in population]

        pareto_front = []
        pareto_objectives = []

        for i, obj_i in enumerate(objectives):
            dominated = False
            for j, obj_j in enumerate(objectives):
                if i != j and self.dominates(obj_j, obj_i):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(population[i])
                pareto_objectives.append(obj_i)

        return pareto_front, pareto_objectives


# Example
mo_fitness = MultiObjectiveFitness(X, y, Ridge(alpha=1.0))

# Generate random population
import random
population = [
    [random.randint(0, 1) for _ in range(50)]
    for _ in range(100)
]

# Get Pareto front
pareto_front, pareto_objectives = mo_fitness.get_pareto_front(population)

print(f"Pareto front size: {len(pareto_front)}")
print("\nPareto-optimal solutions:")
for ind, obj in zip(pareto_front[:5], pareto_objectives[:5]):
    print(f"  Features: {sum(ind):2d}, Accuracy: {obj[0]:.4f}")
```

## Regularized Fitness

Incorporate L1/L2 regularization concepts:

```python
class RegularizedFitness:
    """
    Fitness with explicit regularization on feature weights.
    """

    def __init__(self, X, y, cv=5, alpha_l1=0.01, alpha_l2=0.01):
        self.X = np.array(X)
        self.y = np.array(y)
        self.cv = cv
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.n_features = X.shape[1]

    def evaluate(self, chromosome):
        """
        Fitness = CV_score - L1_penalty - L2_penalty
        """
        selected = [i for i, x in enumerate(chromosome) if x == 1]

        if len(selected) == 0:
            return -np.inf

        X_subset = self.X[:, selected]

        # Fit model and get coefficients
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()

        # CV score
        scores = cross_val_score(
            model, X_subset, self.y,
            cv=self.cv, scoring='neg_mean_squared_error'
        )
        cv_score = scores.mean()

        # Fit on full data to get coefficients
        model.fit(X_subset, self.y)
        coefficients = model.coef_

        # Regularization penalties
        l1_penalty = self.alpha_l1 * np.sum(np.abs(coefficients))
        l2_penalty = self.alpha_l2 * np.sum(coefficients ** 2)

        fitness = cv_score - l1_penalty - l2_penalty

        return fitness


reg_fitness = RegularizedFitness(X, y, alpha_l1=0.001, alpha_l2=0.001)
fitness = reg_fitness.evaluate([1] * 10 + [0] * 40)
print(f"Regularized fitness: {fitness:.4f}")
```

## Caching and Efficiency

```python
class EfficientFitnessEvaluator:
    """
    Fitness evaluator with advanced caching and efficiency features.
    """

    def __init__(self, X, y, model, cv=5, metric='neg_mse',
                 cache_size=10000):
        self.X = np.array(X)
        self.y = np.array(y)
        self.model = model
        self.cv = cv
        self.metric = metric
        self.n_features = X.shape[1]

        # LRU-style cache
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.cache_size = cache_size

        # Statistics
        self.stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def evaluate(self, chromosome):
        """Evaluate with caching."""
        self.stats['evaluations'] += 1

        # Convert to hashable key
        key = tuple(chromosome)

        if key in self.cache:
            self.stats['cache_hits'] += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]

        self.stats['cache_misses'] += 1

        # Compute fitness
        selected = [i for i, x in enumerate(chromosome) if x == 1]

        if len(selected) == 0:
            fitness = -np.inf
        else:
            X_subset = self.X[:, selected]
            try:
                scores = cross_val_score(
                    self.model, X_subset, self.y,
                    cv=self.cv, scoring=self.metric
                )
                fitness = scores.mean()
            except:
                fitness = -np.inf

        # Add to cache
        self.cache[key] = fitness

        # Evict if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return fitness

    def evaluate_batch(self, population):
        """
        Evaluate multiple individuals efficiently.
        """
        # Separate cached and uncached
        results = {}
        to_evaluate = []

        for ind in population:
            key = tuple(ind)
            if key in self.cache:
                results[key] = self.cache[key]
            else:
                to_evaluate.append(ind)

        # Evaluate uncached
        for ind in to_evaluate:
            fitness = self.evaluate(ind)
            results[tuple(ind)] = fitness

        return [results[tuple(ind)] for ind in population]

    def get_statistics(self):
        """Return evaluation statistics."""
        cache_hit_rate = (
            self.stats['cache_hits'] / max(self.stats['evaluations'], 1)
        )
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache)
        }


# Example usage
efficient_evaluator = EfficientFitnessEvaluator(X, y, Ridge(alpha=1.0))

# Simulate GA evaluations
population = [[random.randint(0, 1) for _ in range(50)] for _ in range(50)]

for generation in range(10):
    # Evaluate (some will be cached from previous generations)
    fitnesses = efficient_evaluator.evaluate_batch(population)

    # Create new population (with some overlap)
    new_pop = population[:25] + [[random.randint(0, 1) for _ in range(50)] for _ in range(25)]
    population = new_pop

print("\nEfficient Evaluator Statistics:")
stats = efficient_evaluator.get_statistics()
for k, v in stats.items():
    print(f"  {k}: {v}")
```

## Key Takeaways

<div class="callout-key">
🔑 **Key Points**

1. **Cross-validation** provides robust out-of-sample fitness estimates

2. **Time series CV** prevents lookahead bias in financial applications

3. **Feature penalties** balance accuracy vs. parsimony

4. **Multi-objective** approaches find the Pareto frontier of solutions

5. **Caching** dramatically speeds up GA with repeated evaluations

6. **Regularization** can be incorporated into fitness for smoother landscapes
</div>
---

**Next:** [Companion Slides](./02_cross_validation_fitness_slides.md) | [Notebook](../notebooks/01_fitness_functions.ipynb)

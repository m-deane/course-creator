# Module 5: Advanced Topics

## Overview

Explore advanced GA techniques: multi-objective optimization, hybrid methods, and specialized operators for feature selection.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Implement multi-objective feature selection
2. Combine GAs with local search
3. Design adaptive operators
4. Compare with other metaheuristics

## Contents

### Guides
- `01_multi_objective.md` - NSGA-II for feature selection
- `02_hybrid_methods.md` - GA + local search
- `03_adaptive_operators.md` - Self-adjusting GAs

### Notebooks
- `01_nsga2_features.ipynb` - Multi-objective selection
- `02_hybrid_ga.ipynb` - Memetic algorithms

## Key Concepts

### Multi-Objective Feature Selection

Optimize simultaneously:
- Minimize prediction error
- Minimize number of features
- Maximize feature diversity

```python
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))

def evaluate_multi(individual, X, y):
    accuracy = cv_score(individual, X, y)
    complexity = sum(individual) / len(individual)
    return (accuracy, complexity)

# Use NSGA-II
toolbox.register("select", tools.selNSGA2)
```

### Hybrid Methods (Memetic)

```
GA Evolution → Local Search → GA Evolution → ...

1. Evolve population with GA
2. Apply local search to best individuals
3. Replace with improved solutions
4. Continue evolution
```

### Adaptive Parameters

```python
class AdaptiveGA:
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def adapt(self, diversity, improvement):
        if diversity < 0.1:
            self.mutation_rate *= 1.5  # Increase exploration
        if improvement < 0.01:
            self.crossover_rate *= 0.9  # Change exploitation
```

### Comparison Methods

| Method | Strengths | When to Use |
|--------|-----------|-------------|
| GA | Global search, flexible | Large feature spaces |
| PSO | Faster convergence | Continuous features |
| SA | Simple, good local | Smaller problems |
| ACO | Path-based | Ordered features |

## Prerequisites

- Module 0-4 completed
- Optimization background

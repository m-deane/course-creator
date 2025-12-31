# Module 4: Implementation with DEAP

## Overview

Implement feature selection GAs using DEAP (Distributed Evolutionary Algorithms in Python). Build production-ready, parallelized implementations.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Use DEAP for GA implementation
2. Customize operators for feature selection
3. Parallelize fitness evaluation
4. Log and analyze evolution

## Contents

### Guides
- `01_deap_basics.md` - DEAP framework overview
- `02_custom_operators.md` - Feature selection operators
- `03_parallelization.md` - Multi-core execution

### Notebooks
- `01_deap_ga.ipynb` - Complete DEAP implementation
- `02_parallel_evolution.ipynb` - Scalable GA

## Key Concepts

### DEAP Setup

```python
from deap import base, creator, tools, algorithms
import numpy as np

# Create fitness and individual types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize toolbox
toolbox = base.Toolbox()

# Attribute generator (binary)
toolbox.register("attr_bool", np.random.randint, 0, 2)

# Individual and population
n_features = 50
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
```

### Register Operators

```python
# Evaluation function
def evaluate(individual, X, y):
    if sum(individual) == 0:
        return (-1e6,)  # Penalize empty selection

    selected = X[:, np.array(individual) == 1]
    score = cross_val_fitness(selected, y)
    return (score,)

toolbox.register("evaluate", evaluate, X=X, y=y)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
```

### Run Evolution

```python
# Create population
pop = toolbox.population(n=100)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

# Hall of Fame
hof = tools.HallOfFame(5)

# Run GA
pop, log = algorithms.eaSimple(
    pop, toolbox,
    cxpb=0.7, mutpb=0.2,
    ngen=50,
    stats=stats, halloffame=hof,
    verbose=True
)
```

### Parallelization

```python
from multiprocessing import Pool

# Parallel fitness evaluation
pool = Pool(processes=4)
toolbox.register("map", pool.map)

# Now evaluation runs in parallel
```

## Prerequisites

- Module 0-3 completed
- Python proficiency

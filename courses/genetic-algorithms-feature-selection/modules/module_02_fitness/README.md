# Module 2: Fitness Function Design

## Overview

Design fitness functions that accurately evaluate feature subsets while preventing overfitting. The fitness function is the heart of feature selection with GAs.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Design fitness functions for forecasting
2. Implement cross-validation in fitness
3. Add parsimony pressure to prevent overfitting
4. Balance multiple objectives

## Contents

### Guides
- `01_fitness_design.md` - Principles of fitness functions
- `02_validation_strategies.md` - CV in fitness evaluation
- `03_multi_objective.md` - Accuracy vs complexity

### Notebooks
- `01_fitness_functions.ipynb` - Implementing fitness
- `02_overfitting_prevention.ipynb` - Validation experiments

## Key Concepts

### Basic Fitness Function

```python
def fitness(chromosome, X, y, model):
    # Select features
    selected = X[:, chromosome == 1]

    # Cross-validation
    scores = cross_val_score(model, selected, y, cv=5, scoring='neg_mse')

    # Return negative MSE (higher is better)
    return np.mean(scores)
```

### Validation Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| K-Fold | Random splits | Cross-sectional |
| Walk-Forward | Time-ordered | Time series |
| Nested CV | Outer/inner | Hyperparameter selection |
| Bootstrap | Resampling | Confidence intervals |

### Parsimony Pressure

$$\text{Fitness} = \text{Accuracy} - \lambda \cdot \text{Complexity}$$

Where complexity = number of selected features / total features

### Multi-Objective Fitness

```python
def multi_objective_fitness(chromosome, X, y):
    accuracy = -cv_mse(chromosome, X, y)
    complexity = sum(chromosome) / len(chromosome)

    return (accuracy, -complexity)  # Maximize both
```

## Prerequisites

- Module 0-1 completed
- Cross-validation understanding

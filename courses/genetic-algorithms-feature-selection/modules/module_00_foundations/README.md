# Module 0: Foundations

## Overview

Establish the foundations of optimization and search for feature selection. Review the feature selection problem and why evolutionary approaches excel.

**Time Estimate:** 4-6 hours

## Learning Objectives

By completing this module, you will:
1. Understand the feature selection problem
2. Compare selection approaches (filter, wrapper, embedded)
3. Recognize when GAs are appropriate
4. Set up your computational environment

## Contents

### Guides
- `01_feature_selection_problem.md` - The curse of dimensionality
- `02_selection_approaches.md` - Filter, wrapper, embedded
- `03_optimization_basics.md` - Search spaces and objectives

### Notebooks
- `01_selection_comparison.ipynb` - Comparing approaches
- `02_environment_setup.ipynb` - DEAP and tools

## Key Concepts

### The Feature Selection Problem

Given $p$ candidate features, there are $2^p$ possible subsets:

| Features | Possible Subsets |
|----------|-----------------|
| 10 | 1,024 |
| 20 | 1,048,576 |
| 50 | 1.1 × 10^15 |
| 100 | 1.3 × 10^30 |

Exhaustive search is infeasible for $p > 20$.

### Selection Approaches

| Approach | Description | Computational Cost |
|----------|-------------|-------------------|
| Filter | Statistical metrics | Low |
| Wrapper | Model-based evaluation | High |
| Embedded | Built into model | Medium |
| GA | Evolutionary search | Medium-High |

### Why Genetic Algorithms?

- Handle large search spaces
- No gradient required
- Find diverse solutions
- Naturally parallel
- Flexible fitness functions

## Prerequisites

- Python proficiency
- ML model basics
- Time series fundamentals

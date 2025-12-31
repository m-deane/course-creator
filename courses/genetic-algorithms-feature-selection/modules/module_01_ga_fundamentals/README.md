# Module 1: GA Fundamentals

## Overview

Master the core components of genetic algorithms: chromosome encoding, selection, crossover, and mutation operators.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. Design chromosome representations for features
2. Implement selection operators
3. Apply crossover and mutation
4. Build a basic GA from scratch

## Contents

### Guides
- `01_encoding.md` - Binary and real-valued chromosomes
- `02_selection.md` - Tournament, roulette, rank selection
- `03_genetic_operators.md` - Crossover and mutation

### Notebooks
- `01_basic_ga.ipynb` - GA from scratch
- `02_operator_analysis.ipynb` - Comparing operators

## Key Concepts

### Binary Encoding for Features

```
Features:   [lag1, lag2, sma10, sma20, rsi, macd, vol]
Chromosome: [  1,    0,    1,     1,    0,    1,   1 ]
                ↓
Selected:   [lag1, sma10, sma20, macd, vol]
```

### Selection Operators

| Operator | Description | Selection Pressure |
|----------|-------------|-------------------|
| Tournament | Best of k random | Adjustable (k) |
| Roulette | Proportional to fitness | Low |
| Rank | Proportional to rank | Medium |
| Elitism | Keep best n | Preserves best |

### Crossover Operators

```
Single-Point:
Parent 1: [1,0,1,1|0,0,1]    Child 1: [1,0,1,1|1,1,0]
Parent 2: [0,1,0,0|1,1,0]    Child 2: [0,1,0,0|0,0,1]

Uniform:
Parent 1: [1,0,1,1,0,0,1]    Mask: [1,0,1,0,1,0,1]
Parent 2: [0,1,0,0,1,1,0]    Child: [1,1,1,0,0,1,1]
```

### Mutation

Bit-flip with probability $p_m$:
- Too low: Stagnation
- Too high: Random search
- Typical: 1/n to 0.01

## Prerequisites

- Module 0 completed
- Basic probability

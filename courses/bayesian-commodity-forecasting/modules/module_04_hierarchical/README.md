# Module 4: Hierarchical Models for Related Commodities

## Overview

Commodities don't exist in isolation. Crude oil affects gasoline and heating oil. Corn influences soybeans and wheat. Hierarchical Bayesian models leverage these relationships through partial pooling, improving forecasts by sharing information across related markets.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. **Understand** partial pooling and shrinkage estimation
2. **Build** hierarchical models for commodity complexes
3. **Interpret** group-level and individual-level parameters
4. **Apply** hierarchical priors for improved out-of-sample forecasts
5. **Evaluate** the degree of pooling and its implications

## Module Contents

### Guides
- `01_partial_pooling.md` - The logic of hierarchical models
- `02_energy_complex.md` - Modeling crude, gasoline, heating oil together
- `03_agricultural_complex.md` - Grains and oilseeds relationships

### Notebooks
- `01_pooling_comparison.ipynb` - No pooling vs complete pooling vs partial
- `02_energy_complex_model.ipynb` - Hierarchical model for petroleum products
- `03_agricultural_model.ipynb` - Corn-soy-wheat hierarchical model
- `04_cross_commodity_forecasts.ipynb` - Leveraging cross-market information

### Assessments
- `quiz.md` - Hierarchical model concepts (15 questions)
- `mini_project_rubric.md` - Build hierarchical model for commodity complex

## Key Concepts

### The Pooling Spectrum

```
No Pooling                    Partial Pooling                    Complete Pooling
(Separate models)             (Hierarchical)                     (Single model)
     │                              │                                  │
     │    Each commodity            │    Information shared            │    All commodities
     │    estimated                 │    across commodities            │    treated
     │    independently             │    via group prior               │    identically
     │                              │                                  │
     ▼                              ▼                                  ▼
High variance                 Balanced bias-variance              High bias
Low bias                      Adaptive shrinkage                  Low variance
```

### Why Hierarchical for Commodities?

1. **Energy Complex:** Crude oil price drives product prices through refining margins
2. **Grains:** Corn and soybeans compete for acreage; weather affects all
3. **Metals:** Base metals respond similarly to China demand
4. **Sparse Data:** New contracts or markets benefit from pooling with established ones

### Model Structure

```
Hyperprior Level (Group)
    μ_group ~ Normal(0, σ_group)
    σ_group ~ HalfNormal(...)
         │
         ▼
Prior Level (Individual Commodities)
    β_i ~ Normal(μ_group, σ_group)
         │
         ▼
Likelihood Level
    y_i ~ Normal(X_i β_i, σ_i)
```

## Commodity Complexes

### Energy Complex
| Product | Relationship to Crude |
|---------|----------------------|
| Gasoline | ~0.42 barrels per barrel crude |
| Heating Oil | ~0.25 barrels per barrel crude |
| Jet Fuel | ~0.10 barrels per barrel crude |

**Crack Spread:** Price of products minus price of crude (refining margin)

### Agricultural Complex
| Crop | Relationship |
|------|--------------|
| Corn-Soybean | Compete for acreage |
| Soybean-Meal-Oil | Crush relationship |
| Wheat-Corn | Feed substitution |

### Metals Complex
| Metal | Relationship |
|-------|--------------|
| Copper-Aluminum | Industrial demand |
| Gold-Silver | Monetary/safe haven |
| Platinum-Palladium | Automotive catalysts |

## Completion Criteria

- [ ] Pooling comparison notebook demonstrates shrinkage
- [ ] Energy complex model converges with good diagnostics
- [ ] Mini-project: Hierarchical model for chosen complex
- [ ] Quiz score ≥ 80%

## Prerequisites

- Module 1-3 completed
- PyMC model building experience
- Understanding of regression coefficients

---

*"In hierarchical models, each commodity learns not just from its own data, but from the family it belongs to."*

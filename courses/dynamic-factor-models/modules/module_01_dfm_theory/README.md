# Module 01: Dynamic Factor Model Theory

## Overview

Learn how dynamic factor models (DFMs) extend state space models to extract common patterns from high-dimensional data. In 2 hours, you'll understand why DFMs dominate macroeconomic forecasting and build your first factor extraction system.

**Time Commitment:** 2 hours
**Difficulty:** Intermediate
**Prerequisites:** Module 00 (State Space Models & Kalman Filter)

## Why This Matters

Dynamic factor models power:
- **Central bank nowcasting** - Fed, ECB, BoE use DFMs to track economy in real-time
- **Financial risk systems** - Extract market-wide factors from 1000+ assets
- **Climate modeling** - Common patterns across hundreds of weather stations
- **Supply chain analytics** - Shared dynamics across product categories

DFMs reduce 100+ time series to 3-5 interpretable factors without losing information.

## Learning Objectives

By the end of this module, you will:

1. **Explain** how DFMs differ from static PCA
2. **Specify** a dynamic factor model in state space form
3. **Solve** the identification problem using restrictions
4. **Extract** common factors from real macroeconomic data
5. **Interpret** factor loadings and factor dynamics

## Module Contents

### Guides (Read First)
1. **[Factor Models Overview](guides/01_factor_models_overview.md)** - From PCA to dynamic factors (20 min)
2. **[DFM Specification](guides/02_dfm_specification.md)** - State space representation (25 min)
3. **[Identification](guides/03_identification.md)** - Solving the rotation problem (20 min)
4. **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for DFM formulas

### Notebooks (Hands-On)
1. **[From PCA to DFM](notebooks/01_from_pca_to_dfm.ipynb)** - See why dynamics matter (15 min)
2. **[DFM Specification](notebooks/02_dfm_specification.ipynb)** - Build your first DFM (15 min)

### Practice
- **[Self-Check Exercises](exercises/exercises.py)** - Test your understanding (ungraded)

### Resources
- **[Additional Readings](resources/additional_readings.md)** - Papers on DFM theory and applications
- **[Figures](resources/figures/)** - Visual assets and diagrams

## Recommended Path

### Fast Track (1 hour)
1. Skim [Factor Models Overview](guides/01_factor_models_overview.md)
2. Run [From PCA to DFM notebook](notebooks/01_from_pca_to_dfm.ipynb)
3. Run [DFM Specification notebook](notebooks/02_dfm_specification.ipynb)
4. Check [Cheatsheet](guides/cheatsheet.md)

### Deep Dive (2-3 hours)
1. Read all three guides thoroughly
2. Work through both notebooks, experimenting with parameters
3. Complete self-check exercises
4. Read Stock & Watson (2002) from additional readings
5. Apply to your own data

### Portfolio Extension
Build a real-time nowcasting dashboard using DFM (see Module 03: Applications).

## Key Concepts

- **Factor Structure** - Many variables driven by few common shocks
- **Dynamic Loadings** - Factors affect variables with lags (impulse responses)
- **Identification** - Normalizations required to pin down factors uniquely
- **Observable vs Latent** - Distinction between measured variables and hidden factors

## Common Questions

**Q: How is this different from PCA?**
A: PCA finds static factors (contemporaneous only). DFMs capture dynamics (how factors evolve over time and affect variables with lags).

**Q: How many factors do I need?**
A: Information criteria (IC) or scree plots. Usually 1-5 factors for macro data, 5-15 for financial data.

**Q: Why do we need identification restrictions?**
A: Factor models are rotation-invariant. Without restrictions, infinitely many equivalent representations exist. We need to pin down a unique solution.

**Q: Can DFMs handle mixed frequencies?**
A: Yes! That's one of their superpowers. See Module 05 on mixed-frequency models.

## Next Steps

After completing this module:
- **Module 02:** Learn estimation methods (maximum likelihood, EM algorithm, Bayesian)
- **Module 03:** Apply DFMs to nowcasting, forecasting, and structural analysis
- **Project 1:** Build a multi-indicator nowcasting system

## Conceptual Roadmap

```
Module 00: State Space Models
    └─→ Linear dynamics + Kalman filter
        │
        ▼
Module 01: Dynamic Factor Models (YOU ARE HERE)
    └─→ High-dimensional extension
        │  • Factor structure
        │  • Identification
        │  • State space form
        ▼
Module 02: Estimation
    └─→ How to estimate DFM parameters
        │  • Maximum likelihood
        │  • EM algorithm
        │  • Bayesian methods
        ▼
Module 03: Applications
    └─→ Real-world use cases
```

## Prerequisites Check

Before starting, ensure you understand:
- [x] State space model structure (Module 00)
- [x] Kalman filter (Module 00)
- [x] Basic PCA (eigenvectors, loadings, scores)
- [x] Matrix algebra (eigendecomposition)

If rusty on PCA, review Section 1 of the Factor Models Overview guide.

## Getting Help

- Check [Common Pitfalls](guides/02_dfm_specification.md#common-pitfalls) sections
- Review [Cheatsheet](guides/cheatsheet.md) for quick formulas
- Consult [Additional Readings](resources/additional_readings.md) for deeper theory
- Compare your results to notebook solutions

---

*Dynamic factor models are the workhorse of modern macroeconomic analysis. Master them here.*

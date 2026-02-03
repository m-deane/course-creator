# Module 1: Static Factor Models

## Overview

This module introduces the static factor model, the foundation for all factor-based methods in econometrics. You'll learn the model specification, identification problem, and how to estimate factors using principal components.

**Estimated Time:** 6-8 hours
**Prerequisites:** Module 0 (matrix algebra, PCA basics)

## Learning Objectives

By completing this module, you will be able to:

1. **Formulate** the static factor model in matrix notation
2. **Explain** the identification problem and standard normalization constraints
3. **Distinguish** between exact and approximate factor models
4. **Extract** factors using principal components
5. **Interpret** factor loadings in economic terms

## Module Contents

### Guides
1. `guides/01_factor_model_specification.md` - Model setup and assumptions
2. `guides/02_identification_problem.md` - Why factors aren't unique and how to fix it
3. `guides/03_approximate_factor_models.md` - Large-N theory and weak dependence

### Notebooks
1. `notebooks/01_static_factor_basics.ipynb` - Implement factor model from scratch

### Assessments
- `assessments/quiz_module_01.md` - Conceptual quiz

## Key Concepts

### The Static Factor Model

$$X_{it} = \lambda_i' F_t + e_{it}$$

In matrix form for all $N$ variables at time $t$:

$$X_t = \Lambda F_t + e_t$$

where:
- $X_t$: $N \times 1$ vector of observed variables
- $F_t$: $r \times 1$ vector of latent factors
- $\Lambda$: $N \times r$ matrix of factor loadings
- $e_t$: $N \times 1$ vector of idiosyncratic errors

### The Identification Problem

The model is invariant to invertible transformations:
$$X_t = \Lambda F_t = (\Lambda H)(H^{-1}F_t) = \tilde{\Lambda}\tilde{F}_t$$

Standard normalizations:
- $F'F/T = I_r$ (orthonormal factors)
- $\Lambda'\Lambda$ diagonal (ordered loadings)

### Approximate vs Exact Factor Models

| Aspect | Exact | Approximate |
|--------|-------|-------------|
| Idiosyncratic correlation | None ($E[e_{it}e_{jt}] = 0$) | Weak (bounded) |
| Cross-sectional dependence | Only through factors | Factors + weak local |
| Large-N asymptotics | Not needed | Essential |
| Real-world relevance | Restrictive | More realistic |

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 0 | PCA as factor extraction method |
| Module 2 | Static → Dynamic by adding factor dynamics |
| Module 3 | PCA estimation of static model |
| Module 6 | Factor-augmented regression uses static factors |

## Reading List

### Required
- Bai, J. & Ng, S. (2008). "Large Dimensional Factor Analysis." *Foundations and Trends* §1-2.

### Recommended
- Stock, J.H. & Watson, M.W. (2016). "Dynamic Factor Models..." *Handbook of Macroeconomics* §2.

### Historical Context
- Chamberlain, G. & Rothschild, M. (1983). "Arbitrage, Factor Structure, and Mean-Variance Analysis on Large Asset Markets." *Econometrica*.

## Practical Applications

By the end of this module, you'll be able to:

1. Extract "real activity" and "inflation" factors from macroeconomic panels
2. Decompose variable variance into common and idiosyncratic components
3. Evaluate how well factors explain cross-sectional co-movement
4. Visualize and interpret factor loadings

## Next Steps

After completing this module:
1. Verify understanding with the conceptual quiz
2. Complete the FRED-MD factor extraction notebook
3. Proceed to Module 2: Dynamic Factor Models

---

*"The factor model is one of the great inventions of statistics... it has been incredibly fruitful in economics." - James Stock*

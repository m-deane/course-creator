# Module 04: Advanced Extensions

## Overview

This module explores cutting-edge extensions of Dynamic Factor Models: **time-varying parameters** (adapting to structural breaks), **mixed-frequency data** (combining monthly and quarterly variables), and **high-dimensional methods** (handling hundreds of predictors). These advanced techniques are essential for modern macroeconomic forecasting and adapt DFMs to real-world data challenges.

**Duration:** 2 weeks
**Effort:** 8-10 hours per week

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement** time-varying parameter DFMs for structural break adaptation
2. **Estimate** mixed-frequency DFMs combining monthly and quarterly data
3. **Apply** MIDAS regression and bridge equation approaches to temporal aggregation
4. **Handle** high-dimensional datasets (N > 100) using regularization and dimension reduction
5. **Compare** full-system vs two-step estimation methods for computational efficiency
6. **Diagnose** when parameter variation matters vs constant-parameter approximations
7. **Build** production-ready systems handling multiple data frequencies

## Why This Matters

**Real-world challenges:**
- **Structural breaks:** COVID-19, financial crisis, regime shifts → constant parameters fail
- **Mixed frequencies:** GDP is quarterly, but IP, employment are monthly → how to combine?
- **High dimensionality:** Modern datasets have 100+ series → standard MLE breaks down
- **Computational limits:** Full-system MLE with N=200, T=300 takes hours → need fast methods

**Practical applications:**
- Central banks use time-varying DFMs to detect structural breaks in real-time
- Mixed-frequency models enable nowcasting with arbitrary publication lags
- Sparse methods identify which 10 of 200 predictors actually matter

## Module Contents

### Guides (Detailed Concept Explanations)

1. **[Time-Varying Parameters](guides/01_time_varying_parameters.md)** - Adaptive DFMs
   - Random walk parameters vs structural breaks
   - State-space formulation with stochastic loadings
   - Forgetting factors and rolling windows
   - Online learning and recursive estimation

2. **[Mixed-Frequency Models](guides/02_mixed_frequency.md)** - MIDAS and bridge equations
   - Temporal aggregation theory
   - MIDAS regression (distributed lags)
   - State-space mixed-frequency DFM
   - Skip-sampling vs bridge equations

3. **[Large Datasets](guides/03_large_datasets.md)** - High-dimensional DFMs
   - Two-step estimation (PCA + Kalman)
   - Targeted predictors and pre-selection
   - Sparse factor models (LASSO loadings)
   - Computational tricks (woodbury identity, rank reduction)

4. **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for module concepts

### Notebooks (15-Minute Hands-On)

1. **[Time-Varying Parameter DFM](notebooks/01_tvp_dfm.ipynb)** - Detect structural breaks
   - Simulate DFM with parameter shift
   - Implement rolling-window estimation
   - Compare constant vs time-varying parameters
   - Visualize loading evolution over time

2. **[Mixed-Frequency DFM](notebooks/02_mixed_frequency.ipynb)** - Combine monthly + quarterly
   - Download mixed-frequency FRED data
   - Implement MIDAS regression
   - State-space mixed-frequency DFM
   - Compare to quarterly-only model

### Exercises (Self-Check, Ungraded)

**[exercises.py](exercises/exercises.py)** - Practice problems with instant feedback
- Implement time-varying Kalman filter
- Build MIDAS bridge equation
- Two-step DFM estimation
- High-dimensional factor number selection

### Resources

- **[Additional Readings](resources/additional_readings.md)** - Curated papers and documentation
- **[Figures](resources/figures/)** - Visual assets and diagrams

## Prerequisites

**From previous modules:**
- Kalman filter (Module 2)
- Nowcasting framework (Module 3)
- Missing data handling (Module 3)

**Additional concepts:**
- Structural breaks in time series
- MIDAS regression basics
- Regularization (LASSO, Ridge)
- Computational complexity (Big-O notation helpful)

## Key Innovations

### Time-Varying Parameters
**Problem:** Post-COVID, pre-COVID factor loadings differ → constant-parameter DFM fails
**Solution:** Allow loadings Λ_t to evolve over time
**Tradeoff:** More flexible but requires more data (estimate T×N×r parameters!)

### Mixed-Frequency
**Problem:** GDP is quarterly, employment is monthly → information loss if aggregating up
**Solution:** Model monthly factors, map to quarterly GDP via skip-sampling
**Advantage:** Use all monthly data, no pre-aggregation

### High-Dimensional
**Problem:** N=200 series, full MLE has 200² parameters in Σ_e → overfitting
**Solution:** Two-step (fast PCA) or sparse methods (regularize loadings)
**Speedup:** 100x faster than full MLE for N > 50

## Practical Skills You'll Gain

1. **Structural Break Detection**
   - Rolling window estimation
   - Recursive forecast comparison
   - Chow tests for parameter stability
   - Bayesian change-point detection

2. **Mixed-Frequency Pipelines**
   - Skip-sampling alignment
   - MIDAS distributed lag selection
   - State-space formulation
   - Ragged edge with multiple frequencies

3. **High-Dimensional Estimation**
   - Two-step PCA-Kalman workflow
   - Targeted predictor selection
   - Sparse loadings via LASSO
   - Rank reduction tricks

## Success Criteria

You're ready to move on when you can:

- [ ] Implement rolling-window DFM estimation
- [ ] Detect when time-varying parameters improve forecasts
- [ ] Build mixed-frequency bridge equation (monthly → quarterly)
- [ ] Estimate DFM with N > 100 series efficiently
- [ ] Explain when two-step beats full MLE (and vice versa)
- [ ] Handle COVID-era structural breaks in nowcasting

## Connections

**Builds on:**
- Module 02: Kalman filter, state-space methods
- Module 03: Nowcasting, missing data, forecast evaluation

**Prepares for:**
- Module 05: Factor-augmented regression (FAVAR)
- Module 06: Sparse methods and feature selection
- Capstone: Production nowcasting with all extensions

## Estimated Time Breakdown

| Activity | Time |
|----------|------|
| Read guides | 4 hours |
| Complete notebooks | 4 hours |
| Self-check exercises | 2 hours |
| Additional readings (optional) | 3-4 hours |

## Getting Started

1. Read **[Time-Varying Parameters Guide](guides/01_time_varying_parameters.md)** to understand adaptation
2. Work through **[TVP-DFM Notebook](notebooks/01_tvp_dfm.ipynb)** to implement rolling windows
3. Read **[Mixed-Frequency Guide](guides/02_mixed_frequency.md)** for temporal aggregation
4. Complete **[Mixed-Frequency Notebook](notebooks/02_mixed_frequency.ipynb)** with real data
5. Read **[Large Datasets Guide](guides/03_large_datasets.md)** for computational methods
6. Test yourself with **[exercises.py](exercises/exercises.py)**

## Common Questions

**Q: When should I use time-varying parameters?**
A: When you suspect structural breaks (recessions, policy changes, pandemics). If parameters stable over time, constant-parameter DFM is simpler.

**Q: Why not just aggregate monthly data to quarterly?**
A: You lose information! Monthly data has 3x more observations within each quarter. Mixed-frequency models use all data optimally.

**Q: Is two-step estimation (PCA + Kalman) worse than full MLE?**
A: Asymptotically equivalent for large N. For N > 50, computational gains far outweigh small efficiency loss.

**Q: How many factors for N=200 series?**
A: Typically 5-10. Use information criteria (BIC) or scree plot to select. More factors = overfit risk.

## Advanced Topics (Optional)

- **Bayesian time-varying DFMs:** Prior on parameter evolution
- **Non-linear extensions:** Threshold DFMs, Markov-switching
- **Alternative temporal aggregation:** Stock-flow consistency, partial aggregation
- **Distributed computing:** Parallel Kalman filter for very large N

---

**Next Module:** Factor-Augmented Regression (FAVAR) and forecast combination

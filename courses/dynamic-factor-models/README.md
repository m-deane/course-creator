# Dynamic Factor Models for Time Series Econometrics

## Course Overview

This advanced course provides a comprehensive treatment of Dynamic Factor Models (DFMs), one of the most powerful tools in modern econometrics for handling high-dimensional time series data. You'll learn to extract latent factors from large datasets, build forecasting models that leverage cross-sectional information, and apply these techniques to real-world problems in macroeconomic nowcasting and financial analysis.

**Level:** Graduate / Professional
**Prerequisites:** Linear algebra, probability theory, basic econometrics (OLS, MLE), Python proficiency
**Duration:** 9 modules + capstone (12-14 weeks)
**Estimated Effort:** 8-10 hours per week

## Why Dynamic Factor Models?

Factor models address fundamental challenges in modern data analysis:

1. **Dimensionality Reduction** - Extract meaningful signals from hundreds of economic indicators without overfitting
2. **Nowcasting** - Produce timely estimates of economic conditions before official data releases
3. **Mixed Frequencies** - Combine monthly, weekly, and daily data in a principled framework
4. **Missing Data** - Handle publication lags and ragged-edge data naturally through state-space methods
5. **Forecast Combination** - Optimally aggregate information from many predictors into factor-augmented regressions

## Learning Outcomes

By completing this course, you will be able to:

1. **Explain** the theoretical foundations of static and dynamic factor models, including identification and estimation
2. **Implement** factor extraction using PCA, maximum likelihood, and Bayesian methods from scratch
3. **Build** state-space representations of DFMs and apply Kalman filtering/smoothing
4. **Construct** nowcasting models for GDP, inflation, and other macroeconomic targets
5. **Handle** mixed-frequency data using MIDAS and state-space approaches
6. **Apply** factor-augmented regression (FAVAR) for forecasting and structural analysis
7. **Select** relevant predictors using sparse factor models and penalized estimation
8. **Evaluate** forecast performance using proper scoring rules and real-time vintages

## Course Structure

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| 0 | Foundations | Matrix algebra review, time series basics, PCA refresher |
| 1 | Static Factor Models | Factor analysis, identification, rotation, approximate factors |
| 2 | Dynamic Factor Models | State-space form, Kalman filter, dynamic factor structure |
| 3 | Estimation I: Principal Components | Stock-Watson approach, consistency, factor number selection |
| 4 | Estimation II: Likelihood Methods | MLE, EM algorithm, Bayesian estimation with priors |
| 5 | Mixed Frequency & Nowcasting | MIDAS, bridge equations, ragged-edge data, real-time forecasting |
| 6 | Factor-Augmented Regression | FAVAR, diffusion indices, forecast combination |
| 7 | Sparse Methods & Feature Selection | Targeted predictors, LASSO, elastic net, variable selection |
| 8 | Advanced Topics | Time-varying parameters, non-Gaussian factors, machine learning |
| Capstone | Nowcasting System | End-to-end real-time macroeconomic forecasting |

## Technical Requirements

```bash
# Create environment
conda create -n dfm-course python=3.11
conda activate dfm-course

# Core packages
pip install numpy scipy pandas matplotlib seaborn statsmodels
pip install scikit-learn linearmodels arch

# State-space and Bayesian
pip install pymc arviz numpyro jax jaxlib

# Data access
pip install pandas-datareader fredapi yfinance

# Notebooks and visualization
pip install jupyterlab ipywidgets plotly
```

## Data Sources

- **FRED-MD** - McCracken & Ng monthly macroeconomic database (127 series)
- **FRED-QD** - Quarterly macroeconomic database
- **ALFRED** - Real-time vintage data for nowcasting evaluation
- **Financial data** - Yield curves, equity factors, volatility indices

## Assessment Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Weekly Quizzes | 15% | Conceptual understanding checks |
| Coding Exercises | 25% | Module notebooks with auto-graded tests |
| Mini-Projects | 30% | Bi-weekly applied modeling tasks |
| Capstone | 30% | Real-time nowcasting system |

## Key References

### Foundational Papers
- Stock, J.H. & Watson, M.W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *JASA*
- Bai, J. & Ng, S. (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*
- Giannone, D., Reichlin, L. & Small, D. (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *JME*

### Textbooks
- Stock, J.H. & Watson, M.W. (2016). "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics*
- Bai, J. & Ng, S. (2008). "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics*

## Getting Started

1. Complete environment setup in `resources/environment_setup.md`
2. Run the diagnostic assessment in `module_00_foundations/`
3. Review matrix algebra and time series prerequisites if needed
4. Begin with Module 1 on static factor models

## Connections to Other Courses

- **Complements:** Bayesian Commodity Forecasting (state-space methods)
- **Prerequisites for:** Advanced Macroeconometrics, Financial Econometrics
- **Related:** Panel Regression, Hidden Markov Models

---

*"In God we trust, all others must bring data. But when you have too much data, factor models help you find the signal."*

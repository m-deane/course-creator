# Learning Objectives by Module

## Bloom's Taxonomy Levels Used
- **Remember:** Recall facts and basic concepts
- **Understand:** Explain ideas and concepts
- **Apply:** Use information in new situations
- **Analyze:** Draw connections among ideas
- **Evaluate:** Justify decisions or courses of action
- **Create:** Produce new or original work

---

## Module 0: Foundations & Prerequisites

### Knowledge Objectives
- [ ] **Remember** the axioms of probability and key distributions (Normal, Gamma, Beta)
- [ ] **Understand** the relationship between joint, marginal, and conditional probability
- [ ] **Understand** the basic structure and participants in commodity markets

### Skill Objectives
- [ ] **Apply** Python/NumPy to compute probabilities and simulate distributions
- [ ] **Apply** environment setup procedures to configure the course stack

### Assessment Alignment
- Diagnostic quiz assesses Remember/Understand objectives
- Setup verification confirms Apply objectives

---

## Module 1: Bayesian Fundamentals for Time Series

### Knowledge Objectives
- [ ] **Understand** Bayes' theorem and its components (prior, likelihood, posterior)
- [ ] **Understand** the concept of conjugate priors and when they apply
- [ ] **Analyze** the differences between Bayesian and frequentist inference philosophies

### Skill Objectives
- [ ] **Apply** PyMC to specify and fit simple Bayesian models
- [ ] **Apply** analytical derivations for conjugate posterior updates
- [ ] **Analyze** the effect of prior choice on posterior inference

### Assessment Alignment
- Quiz 1: Conceptual understanding of Bayes' theorem and conjugacy
- Notebook: PyMC implementation of Bayesian regression

---

## Module 2: Commodity Market Data & Features

### Knowledge Objectives
- [ ] **Remember** key commodity data sources (EIA, USDA, CFTC, LME)
- [ ] **Understand** seasonality patterns in agricultural and energy commodities
- [ ] **Understand** futures term structure and roll dynamics

### Skill Objectives
- [ ] **Apply** data APIs to retrieve commodity fundamental data
- [ ] **Apply** seasonality decomposition methods (STL, Fourier)
- [ ] **Create** engineered features from raw commodity data

### Assessment Alignment
- Quiz 2: Data source knowledge and seasonality concepts
- Mini-Project 1: Build commodity data pipeline with feature engineering

---

## Module 3: Bayesian State Space Models

### Knowledge Objectives
- [ ] **Understand** the general state space model formulation
- [ ] **Understand** the Kalman filter as optimal Bayesian inference for linear-Gaussian systems
- [ ] **Analyze** the connection between state space models and ARIMA

### Skill Objectives
- [ ] **Apply** local level and local linear trend models to commodity prices
- [ ] **Apply** stochastic volatility models for uncertainty quantification
- [ ] **Evaluate** model fit using posterior predictive checks

### Assessment Alignment
- Quiz 3: State space theory and Kalman filter mechanics
- Notebook: Implement and compare state space models

---

## Module 4: Hierarchical Models for Related Commodities

### Knowledge Objectives
- [ ] **Understand** partial pooling and its advantages over complete pooling/no pooling
- [ ] **Understand** how hierarchical priors enable information sharing
- [ ] **Analyze** when hierarchical structure is appropriate for commodity modeling

### Skill Objectives
- [ ] **Apply** hierarchical models to related commodities (energy complex, grains)
- [ ] **Apply** shrinkage estimation for improved out-of-sample forecasts
- [ ] **Evaluate** the degree of pooling and its implications

### Assessment Alignment
- Quiz 4: Hierarchical model theory and shrinkage concepts
- Mini-Project 2: Build hierarchical model for commodity complex

---

## Module 5: Gaussian Processes for Price Forecasting

### Knowledge Objectives
- [ ] **Understand** GPs as distributions over functions
- [ ] **Understand** the role of kernel functions in encoding prior beliefs
- [ ] **Analyze** the relationship between GPs and Bayesian linear regression

### Skill Objectives
- [ ] **Apply** GP regression to commodity price forecasting
- [ ] **Create** custom kernels for commodity seasonality (periodic + trend)
- [ ] **Apply** sparse GP approximations for computational efficiency

### Assessment Alignment
- Quiz 5: GP fundamentals and kernel properties
- Notebook: GP forecasting with custom seasonality kernel

---

## Module 6: Inference Algorithms & Diagnostics

### Knowledge Objectives
- [ ] **Understand** MCMC foundations and the Metropolis-Hastings algorithm
- [ ] **Understand** Hamiltonian Monte Carlo and the NUTS sampler
- [ ] **Understand** variational inference as optimization-based approximate inference

### Skill Objectives
- [ ] **Apply** convergence diagnostics (R-hat, ESS, trace plots)
- [ ] **Analyze** when to use MCMC vs. variational inference
- [ ] **Evaluate** posterior quality using diagnostic metrics

### Assessment Alignment
- Quiz 6: Inference algorithm theory and diagnostics interpretation
- Notebook: Compare inference methods and diagnose convergence issues

---

## Module 7: Regime Switching & Structural Breaks

### Knowledge Objectives
- [ ] **Understand** Hidden Markov Models and the forward-backward algorithm
- [ ] **Understand** Bayesian change point detection methods
- [ ] **Analyze** commodity super-cycles and regime characteristics

### Skill Objectives
- [ ] **Apply** Markov-switching models to commodity price dynamics
- [ ] **Apply** change point detection to identify structural breaks
- [ ] **Evaluate** regime identification using posterior regime probabilities

### Assessment Alignment
- Quiz 7: HMM theory and change point concepts
- Mini-Project 3: Build regime detection system for commodity market

---

## Module 8: Fundamentals Integration & Forecast Combination

### Knowledge Objectives
- [ ] **Understand** storage theory and the theory of normal backwardation
- [ ] **Understand** supply/demand balance frameworks for commodities
- [ ] **Understand** Bayesian model averaging principles

### Skill Objectives
- [ ] **Apply** fundamental variables in Bayesian forecasting models
- [ ] **Apply** Bayesian model averaging for forecast combination
- [ ] **Evaluate** forecasts using proper scoring rules (CRPS, log score)

### Assessment Alignment
- Quiz 8: Fundamentals theory and forecast evaluation
- Mini-Project 4: Integrated forecast combination system

---

## Capstone Project

### Knowledge Objectives
- [ ] **Evaluate** the strengths and limitations of different modeling approaches for a specific commodity
- [ ] **Analyze** the key drivers and dynamics of the chosen market

### Skill Objectives
- [ ] **Create** an end-to-end Bayesian forecasting system
- [ ] **Create** documentation suitable for stakeholder communication
- [ ] **Evaluate** system performance through rigorous backtesting

### Assessment Alignment
- Proposal: Market analysis and modeling approach justification
- Checkpoints: Implementation progress and intermediate results
- Final: Complete system with report and presentation

---

## Cross-Cutting Competencies

### Developed Throughout Course
1. **Uncertainty Communication** - Translating probabilistic forecasts into actionable insights
2. **Model Criticism** - Identifying model limitations and failure modes
3. **Computational Thinking** - Balancing statistical rigor with computational constraints
4. **Domain Integration** - Connecting statistical methods to commodity market economics
5. **Reproducibility** - Maintaining documented, version-controlled analysis pipelines

---

## Objective Coverage Matrix

| Module | Remember | Understand | Apply | Analyze | Evaluate | Create |
|--------|----------|------------|-------|---------|----------|--------|
| 0 | High | Medium | Medium | - | - | - |
| 1 | Low | High | High | Medium | - | - |
| 2 | High | Medium | High | - | - | Medium |
| 3 | Low | High | High | Medium | Medium | - |
| 4 | Low | High | High | High | Medium | - |
| 5 | Low | High | High | Medium | - | Medium |
| 6 | Low | High | High | High | High | - |
| 7 | Low | High | High | High | Medium | - |
| 8 | Low | High | High | - | High | - |
| Cap | - | - | - | High | High | High |

# Learning Objectives by Module

## Course-Level Objectives

By the end of this course, students will be able to:

1. **Theoretical Understanding** - Explain the mathematical foundations of factor models, including identification, estimation consistency, and asymptotic properties
2. **Implementation Skills** - Build factor models from scratch and use production libraries for real applications
3. **Applied Competence** - Design and evaluate forecasting systems using factor-based methods
4. **Critical Analysis** - Assess the strengths and limitations of factor approaches in different contexts

---

## Module 0: Foundations & Prerequisites

### Knowledge Objectives
- K0.1: Recall matrix decomposition techniques (eigendecomposition, SVD) and their properties
- K0.2: Define stationarity, ergodicity, and autocovariance functions for time series
- K0.3: Explain Principal Component Analysis as variance maximization and minimum reconstruction error

### Skill Objectives
- S0.1: Compute eigendecompositions and SVD using NumPy
- S0.2: Implement basic time series operations (lags, differences, rolling statistics)
- S0.3: Apply PCA to multivariate data and interpret loadings/scores

### Assessment Criteria
- [ ] Complete diagnostic quiz with >70% accuracy
- [ ] Successfully execute environment setup notebook
- [ ] Demonstrate prerequisite skills in coding exercises

---

## Module 1: Static Factor Models

### Knowledge Objectives
- K1.1: Formulate the static factor model: $X_t = \Lambda F_t + e_t$
- K1.2: Explain the identification problem and standard normalization constraints
- K1.3: Distinguish between exact and approximate factor models
- K1.4: Describe the relationship between factors, loadings, and idiosyncratic errors

### Skill Objectives
- S1.1: Derive the covariance structure implied by a factor model
- S1.2: Implement factor extraction via principal components
- S1.3: Apply rotation methods (varimax, promax) and interpret rotated factors
- S1.4: Estimate factor scores using regression and Bartlett methods

### Application Objectives
- A1.1: Analyze a macroeconomic dataset to identify latent factors
- A1.2: Interpret extracted factors in economic terms (real activity, inflation, etc.)

### Assessment Criteria
- [ ] Correctly specify factor model in matrix notation
- [ ] Implement PCA-based factor extraction achieving benchmark R²
- [ ] Provide economically meaningful interpretation of factors

---

## Module 2: Dynamic Factor Models

### Knowledge Objectives
- K2.1: Formulate the dynamic factor model with factor dynamics: $F_t = \Phi F_{t-1} + \eta_t$
- K2.2: Express DFM in state-space form (measurement and transition equations)
- K2.3: Derive the Kalman filter recursions for linear Gaussian state-space models
- K2.4: Explain the distinction between filtered and smoothed estimates

### Skill Objectives
- S2.1: Convert a DFM specification into state-space matrices
- S2.2: Implement the Kalman filter algorithm from scratch
- S2.3: Implement the Kalman smoother (Rauch-Tung-Striebel) algorithm
- S2.4: Compute the likelihood via prediction error decomposition

### Application Objectives
- A2.1: Estimate a DFM for a panel of economic indicators
- A2.2: Extract smoothed factor estimates and interpret dynamics
- A2.3: Generate multi-step ahead forecasts using the state-space framework

### Assessment Criteria
- [ ] Correctly implement Kalman filter matching statsmodels output
- [ ] Demonstrate understanding of filter vs smoother estimates
- [ ] Produce forecasts with proper uncertainty quantification

---

## Module 3: Estimation I - Principal Components

### Knowledge Objectives
- K3.1: State the Stock-Watson (2002) two-step estimator
- K3.2: Explain consistency of PC estimators as $N, T \to \infty$
- K3.3: Describe information criteria (IC) for determining factor number (Bai-Ng)
- K3.4: Identify conditions under which PC estimates converge to true factors

### Skill Objectives
- S3.1: Implement standardization and PC extraction for large panels
- S3.2: Apply Bai-Ng information criteria to select number of factors
- S3.3: Implement EM algorithm for PCA with missing data
- S3.4: Compute bootstrap confidence intervals for factor estimates

### Application Objectives
- A3.1: Extract factors from FRED-MD (127 series) dataset
- A3.2: Determine optimal factor number using multiple criteria
- A3.3: Handle missing observations in real macroeconomic data

### Assessment Criteria
- [ ] Correctly extract factors from FRED-MD matching published benchmarks
- [ ] Justify factor number choice using information criteria
- [ ] Handle missing data without errors in pipeline

---

## Module 4: Estimation II - Likelihood Methods

### Knowledge Objectives
- K4.1: Formulate the likelihood function for DFMs via Kalman filter
- K4.2: Describe the EM algorithm for factor model estimation
- K4.3: Specify prior distributions for Bayesian DFM estimation
- K4.4: Explain identification in the likelihood/Bayesian framework

### Skill Objectives
- S4.1: Implement quasi-maximum likelihood estimation for DFMs
- S4.2: Code the EM algorithm with E-step (Kalman smoother) and M-step
- S4.3: Build a Bayesian DFM using PyMC or NumPyro
- S4.4: Diagnose MCMC convergence and compute posterior summaries

### Application Objectives
- A4.1: Compare PC and MLE factor estimates on same dataset
- A4.2: Incorporate prior information in Bayesian estimation
- A4.3: Quantify parameter uncertainty using posterior distributions

### Assessment Criteria
- [ ] EM algorithm converges to reasonable likelihood value
- [ ] Bayesian model produces valid posterior samples (R-hat < 1.1)
- [ ] Compare estimation methods and discuss tradeoffs

---

## Module 5: Mixed Frequency & Nowcasting

### Knowledge Objectives
- K5.1: Define temporal aggregation constraints (stock vs flow variables)
- K5.2: Explain MIDAS (Mixed Data Sampling) regression framework
- K5.3: Describe state-space approach to mixed-frequency factor models
- K5.4: Articulate the "ragged edge" problem and solutions

### Skill Objectives
- S5.1: Implement bridge equations linking monthly indicators to quarterly targets
- S5.2: Build MIDAS regressions with various weighting schemes
- S5.3: Specify state-space model handling mixed frequencies
- S5.4: Update nowcasts as new high-frequency data arrives

### Application Objectives
- A5.1: Construct a nowcasting model for quarterly GDP using monthly indicators
- A5.2: Evaluate nowcast accuracy across different information sets
- A5.3: Produce real-time forecasts handling publication lags

### Assessment Criteria
- [ ] Nowcasting model produces reasonable GDP estimates
- [ ] Demonstrate nowcast improvement as information accumulates
- [ ] Properly handle ragged-edge data structure

---

## Module 6: Factor-Augmented Regression

### Knowledge Objectives
- K6.1: Formulate diffusion index forecasting models
- K6.2: Describe Factor-Augmented VAR (FAVAR) specification
- K6.3: Explain structural identification in FAVAR models
- K6.4: Articulate forecast combination principles

### Skill Objectives
- S6.1: Implement diffusion index regressions with extracted factors
- S6.2: Estimate FAVAR models and compute impulse responses
- S6.3: Combine factor-based forecasts with other models
- S6.4: Evaluate forecast accuracy using proper scoring rules

### Application Objectives
- A6.1: Forecast inflation using factor-augmented models
- A6.2: Analyze monetary policy transmission via FAVAR
- A6.3: Compare factor forecasts to benchmark AR and VAR models

### Assessment Criteria
- [ ] FAVAR impulse responses have correct sign and magnitude
- [ ] Factor forecasts improve on AR benchmark for target variable
- [ ] Proper out-of-sample evaluation methodology applied

---

## Module 7: Sparse Methods & Feature Selection

### Knowledge Objectives
- K7.1: Review high-dimensional regression and regularization
- K7.2: Explain targeted predictors approach (Bai-Ng)
- K7.3: Describe LASSO, elastic net, and adaptive penalties
- K7.4: Articulate the three-pass regression filter

### Skill Objectives
- S7.1: Implement soft and hard thresholding for targeted factors
- S7.2: Apply LASSO/elastic net to factor-augmented regressions
- S7.3: Use cross-validation to select regularization parameters
- S7.4: Implement three-pass regression filter from scratch

### Application Objectives
- A7.1: Select relevant predictors from large panel for forecasting
- A7.2: Compare dense vs sparse factor approaches
- A7.3: Build parsimonious forecasting models for macroeconomic targets

### Assessment Criteria
- [ ] Sparse methods produce interpretable predictor selection
- [ ] Cross-validation properly implemented for hyperparameter selection
- [ ] Forecast performance comparison across dense/sparse methods

---

## Module 8: Advanced Topics

### Knowledge Objectives
- K8.1: Describe time-varying parameter factor models
- K8.2: Explain non-Gaussian factor model extensions
- K8.3: Connect factor models to machine learning approaches
- K8.4: Identify current research frontiers and open problems

### Skill Objectives
- S8.1: Implement factor model with time-varying loadings
- S8.2: Apply robust factor methods for heavy-tailed data
- S8.3: Compare factor approaches to neural network embeddings
- S8.4: Read and critically assess current research papers

### Application Objectives
- A8.1: Analyze structural change in factor loadings over time
- A8.2: Evaluate factor model robustness to outliers
- A8.3: Present research paper findings to peers

### Assessment Criteria
- [ ] Time-varying model captures documented structural breaks
- [ ] Research presentation demonstrates critical understanding
- [ ] Connect course methods to current literature

---

## Capstone Project

### Knowledge Objectives
- KC.1: Integrate all course concepts into coherent forecasting system
- KC.2: Explain design choices and tradeoffs in pipeline construction

### Skill Objectives
- SC.1: Build end-to-end nowcasting pipeline from data to forecasts
- SC.2: Implement proper real-time evaluation methodology
- SC.3: Document and present technical system clearly

### Application Objectives
- AC.1: Produce accurate, timely nowcasts for chosen economic indicator
- AC.2: Evaluate performance against benchmarks using vintage data
- AC.3: Discuss limitations and potential improvements

### Assessment Criteria
- [ ] Pipeline runs without errors on new data
- [ ] Out-of-sample evaluation uses proper real-time vintages
- [ ] Documentation enables reproduction of results
- [ ] Presentation clearly communicates methodology and findings

---

## Bloom's Taxonomy Mapping

| Level | Example Verbs | Course Coverage |
|-------|---------------|-----------------|
| Remember | Define, list, recall | K objectives (definitions, formulas) |
| Understand | Explain, describe, interpret | K objectives (explanations, comparisons) |
| Apply | Implement, compute, use | S objectives (coding, calculation) |
| Analyze | Compare, distinguish, examine | A objectives (method comparison) |
| Evaluate | Assess, judge, critique | Research paper analysis, method evaluation |
| Create | Design, build, construct | Capstone project, novel applications |

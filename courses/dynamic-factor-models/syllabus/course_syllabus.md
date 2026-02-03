# Dynamic Factor Models for Time Series Econometrics
## Course Syllabus

### Course Information

**Course Title:** Dynamic Factor Models for Time Series Econometrics
**Course Code:** ECON 7850 / DATA 7850
**Credits:** 4
**Term:** [Current Term]
**Meeting Time:** [Schedule]
**Location:** [Room / Online]

### Instructor Information

**Instructor:** [Name]
**Email:** [Email]
**Office Hours:** [Times]
**Office:** [Location]

---

## Course Description

This course provides rigorous treatment of dynamic factor models, a cornerstone methodology in modern empirical macroeconomics and financial econometrics. Factor models enable researchers to extract latent common components from high-dimensional panel data, addressing the "curse of dimensionality" while preserving information from hundreds of predictors.

The course progresses from classical static factor analysis through modern dynamic specifications, covering both frequentist (PCA, MLE) and Bayesian estimation approaches. Emphasis is placed on practical applications including macroeconomic nowcasting, real-time forecasting with mixed-frequency data, and factor-augmented regression for prediction and structural analysis.

Students will implement all methods from scratch in Python, building intuition for the underlying mechanics before leveraging production libraries. The capstone project requires building a complete real-time nowcasting system for a macroeconomic indicator.

---

## Learning Objectives

Upon successful completion of this course, students will be able to:

### Knowledge Objectives
1. Explain the identification problem in factor models and standard normalizations
2. Describe the relationship between static and dynamic factor representations
3. Compare principal components, maximum likelihood, and Bayesian estimation approaches
4. Articulate the challenges of mixed-frequency data and real-time forecasting

### Skill Objectives
5. Implement PCA-based factor extraction for large panels
6. Build state-space models and apply Kalman filter/smoother algorithms
7. Estimate dynamic factor models using EM algorithm and MCMC
8. Construct nowcasting models handling ragged-edge data
9. Apply sparse/penalized methods for factor-augmented variable selection

### Application Objectives
10. Evaluate forecast accuracy using proper scoring rules and vintage data
11. Build production-ready forecasting pipelines for economic indicators
12. Critically assess factor model applications in published research

---

## Prerequisites

### Required Background
- **Linear Algebra:** Matrix operations, eigendecomposition, SVD
- **Probability & Statistics:** Random vectors, multivariate normal, MLE
- **Econometrics:** OLS, GLS, time series basics (stationarity, AR/MA)
- **Programming:** Python proficiency (NumPy, Pandas, Matplotlib)

### Recommended Background
- State-space models and Kalman filter (will be reviewed)
- Bayesian inference basics
- Experience with economic data sources (FRED)

### Diagnostic Assessment
Complete the Module 0 diagnostic to assess prerequisite knowledge. Remedial materials are provided for those needing review.

---

## Required Materials

### Software
- Python 3.11+ with Anaconda/Miniconda
- Jupyter Lab or VS Code with Jupyter extension
- Git for version control

### Data Access
- FRED API key (free registration at fred.stlouisfed.org)
- Course datasets provided via course repository

### Textbooks (Recommended, Not Required)
- Bai, J. & Ng, S. (2008). "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics* 3(2): 89-163.
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press. (Chapters 13, 17)

### Course Materials
All materials (notebooks, slides, datasets) available through course repository.

---

## Course Schedule

### Module 0: Foundations & Prerequisites (Week 1)
- Matrix algebra review: eigendecomposition, SVD
- Time series refresher: stationarity, autoregressions
- Principal Component Analysis review
- **Assessment:** Diagnostic quiz, environment setup verification

### Module 1: Static Factor Models (Week 2)
- Classical factor analysis: model specification
- Identification: rotation, normalization constraints
- Approximate factor models for large N
- Factor score estimation
- **Assessment:** Quiz 1, Coding exercises

### Module 2: Dynamic Factor Models (Weeks 3-4)
- From static to dynamic: lagged factor loadings
- State-space representation
- Kalman filter derivation and implementation
- Kalman smoother for full-sample inference
- **Assessment:** Quiz 2, Mini-project 1 (Kalman filter from scratch)

### Module 3: Estimation I - Principal Components (Week 5)
- Stock-Watson two-step estimation
- Consistency results for large N, T
- Determining the number of factors: IC criteria, scree plots
- Handling missing data with EM-PCA
- **Assessment:** Quiz 3, Coding exercises

### Module 4: Estimation II - Likelihood Methods (Weeks 6-7)
- Maximum likelihood via EM algorithm
- Identification in likelihood framework
- Bayesian estimation with conjugate priors
- MCMC implementation for DFMs
- **Assessment:** Quiz 4, Mini-project 2 (Bayesian DFM)

### Module 5: Mixed Frequency & Nowcasting (Weeks 8-9)
- Temporal aggregation and mixed frequencies
- MIDAS regression: bridge equations approach
- State-space approach to mixed frequencies
- Ragged-edge data and real-time information
- Nowcasting GDP: case study
- **Assessment:** Quiz 5, Mini-project 3 (GDP nowcasting)

### Module 6: Factor-Augmented Regression (Week 10)
- Diffusion index forecasting
- Factor-Augmented VAR (FAVAR)
- Structural analysis with factors
- Forecast combination and model averaging
- **Assessment:** Quiz 6, Coding exercises

### Module 7: Sparse Methods & Feature Selection (Weeks 11-12)
- High-dimensional regression review
- Targeted predictors: soft vs hard thresholding
- LASSO and elastic net for factor-augmented models
- Three-pass regression filter
- Variable selection for forecasting
- **Assessment:** Quiz 7, Mini-project 4 (Sparse forecasting)

### Module 8: Advanced Topics (Week 13)
- Time-varying factor loadings
- Non-Gaussian and heavy-tailed factors
- Machine learning meets factor models
- Tensor factor models for panel data
- Current research frontiers
- **Assessment:** Quiz 8, Research paper presentation

### Capstone Project (Weeks 14-15)
- Build complete real-time nowcasting system
- Use FRED-MD and real-time vintages
- Implement factor model + forecasting pipeline
- Evaluate using pseudo out-of-sample analysis
- **Assessment:** Project deliverables, presentation

---

## Assessment

### Grade Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Quizzes (8) | 15% | Weekly conceptual assessments |
| Coding Exercises | 25% | Auto-graded notebook exercises |
| Mini-Projects (4) | 30% | Applied modeling assignments |
| Capstone Project | 25% | End-to-end nowcasting system |
| Participation | 5% | Peer review, discussion contributions |

### Grading Scale

| Grade | Percentage |
|-------|------------|
| A | 93-100 |
| A- | 90-92 |
| B+ | 87-89 |
| B | 83-86 |
| B- | 80-82 |
| C+ | 77-79 |
| C | 73-76 |
| C- | 70-72 |
| D | 60-69 |
| F | Below 60 |

### Late Policy
- Coding exercises: 10% penalty per day, maximum 3 days
- Mini-projects: 15% penalty per day, maximum 2 days
- Capstone: No late submissions accepted

### Academic Integrity
All submitted work must be your own. Code sharing is encouraged for learning but not for graded submissions. Cite all external resources. AI assistants may be used for debugging and explanation but not for generating solutions.

---

## Course Policies

### Attendance
Regular participation expected. Recorded lectures available for asynchronous viewing.

### Communication
- Use course forum for questions (benefits all students)
- Email for private matters only
- Response time: 24-48 hours on weekdays

### Accommodations
Students requiring accommodations should contact the instructor within the first two weeks.

### Technology Requirements
- Reliable internet connection
- Computer capable of running Jupyter notebooks
- Webcam/microphone for office hours (optional)

---

## Additional Resources

### Online Resources
- FRED-MD documentation: [research.stlouisfed.org/econ/mccracken/fred-databases](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
- State-space models in statsmodels: [statsmodels.org/stable/statespace.html](https://www.statsmodels.org/stable/statespace.html)

### Software Documentation
- NumPy: numpy.org/doc
- Statsmodels: statsmodels.org
- PyMC: docs.pymc.io

### Supplementary Readings
Provided in each module's resources folder.

---

## Schedule Summary

| Week | Module | Deliverables |
|------|--------|--------------|
| 1 | M0: Foundations | Diagnostic, Setup |
| 2 | M1: Static Factors | Quiz 1, Exercises |
| 3-4 | M2: Dynamic Factors | Quiz 2, Mini-project 1 |
| 5 | M3: PCA Estimation | Quiz 3, Exercises |
| 6-7 | M4: ML/Bayesian | Quiz 4, Mini-project 2 |
| 8-9 | M5: Nowcasting | Quiz 5, Mini-project 3 |
| 10 | M6: FAVAR | Quiz 6, Exercises |
| 11-12 | M7: Sparse Methods | Quiz 7, Mini-project 4 |
| 13 | M8: Advanced | Quiz 8, Presentation |
| 14-15 | Capstone | Final Project |

---

*This syllabus is subject to change. Students will be notified of any modifications.*

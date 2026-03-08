# Advanced Feature Selection for Tabular Data & Time Series Modelling

## Course Overview

Feature selection is not preprocessing — it is modelling. The choice of which variables enter a model determines its generalisation behaviour, computational cost, interpretability, and resistance to distribution shift. Yet most practitioners treat it as a checkbox: compute correlations, drop low-importance features, move on.

This course treats feature selection as a first-class optimisation problem with rigorous mathematical foundations. You will master every major paradigm — statistical filters, information-theoretic criteria, wrapper search, embedded regularisation, evolutionary and swarm intelligence methods, temporal dependency modelling, high-dimensional screening, causal identification, ensemble combination, and production deployment — and learn when each paradigm is the right tool.

**Level:** Advanced / Expert / Cutting-Edge
**Audience:** Senior data scientists, ML engineers, quant researchers, and PhD-level practitioners who have outgrown sklearn's `SelectKBest`
**Prerequisites:** Strong Python (numpy, pandas, scikit-learn), statistical inference (hypothesis testing, MLE, Bayesian basics), supervised ML modelling, and time series fundamentals (stationarity, ACF/PACF)
**Duration:** 12 modules
**Effort:** 10–12 hours per week for 12 weeks

---

## The Mathematical Motivation

Every feature selection method is an instance of the same combinatorial optimisation problem:

$$S^* = \underset{S \subset \mathcal{F}}{\arg\min} \; \mathcal{L}(\mathcal{M}, S) + \lambda |S|$$

where $\mathcal{F} = \{f_1, f_2, \ldots, f_p\}$ is the full feature set, $S$ is a candidate subset, $\mathcal{L}(\mathcal{M}, S)$ is the generalisation loss of model $\mathcal{M}$ trained on $S$, and $\lambda |S|$ penalises cardinality. The search space has $2^p$ elements — infeasible to enumerate for $p > 30$. Every method in this course is a different strategy for navigating that space efficiently.

Filters approximate $\mathcal{L}$ with a cheap proxy statistic and evaluate features independently or in small groups. Wrappers evaluate $\mathcal{L}$ directly using a held-out validation set, incurring model training cost at each step. Embedded methods bake the cardinality penalty $\lambda |S|$ into the model's training objective. Evolutionary and swarm methods treat the binary inclusion vector as a genome or particle and search $2^p$ stochastically. Causal methods replace $\mathcal{L}$ with an interventional objective, selecting features that remain predictive under distribution shift.

Understanding these connections lets you compose methods, diagnose failures, and design custom selectors for novel problems.

---

## Learning Outcomes

By completing this course, you will be able to:

1. **Derive and implement** information-theoretic feature selection criteria — mutual information, conditional MI, mRMR, JMI, CMIM, DISR — from first principles and apply them to both classification and regression tasks.
2. **Design and run** wrapper search strategies including sequential forward/backward floating search, Boruta with shadow features, and beam search, with correct nested cross-validation to prevent leakage.
3. **Apply embedded regularisation** — Lasso, Elastic Net, Group Lasso, stability selection, and the model-X knockoff filter — understanding the FDR control guarantees each provides.
4. **Implement genetic algorithms and advanced evolutionary methods** (PSO, DE, ACO, NSGA-II) for multi-objective feature selection using DEAP, tuning operators and population dynamics for high-dimensional problems.
5. **Select features correctly in time series contexts** — respecting temporal order in walk-forward validation, applying purged cross-validation with embargo gaps, detecting Granger causality, and handling regime-dependent relevance.
6. **Screen ultra-high-dimensional data** using sure independence screening (SIS/ISIS), random projection, sparse PCA, and estimation-of-distribution algorithms scaled to $p \gg n$ settings.
7. **Identify causal features** using the PC algorithm, FCI, GES, invariant causal prediction (ICP), and double ML with instrumental variables, distinguishing predictive from causal relevance.
8. **Combine selection methods** through ensemble voting, stability aggregation, and multi-criteria Pareto ranking, and quantify selector agreement with Jaccard similarity and Kuncheva's consistency index.
9. **Build production feature selection pipelines** with MLflow experiment tracking, feature store integration, drift monitoring (PSI, KS test), and automated re-selection triggers.
10. **Choose the right method** for any problem by mapping data characteristics — sample size, dimensionality, temporal structure, noise level, interpretability requirements — to the appropriate paradigm.

---

## Module Structure

| Module | Title | Core Methods |
|--------|-------|-------------|
| 0 | The Feature Selection Landscape | Taxonomy, problem formulation, benchmark datasets |
| 1 | Statistical Filter Methods — Beyond Correlation | F-test, HSIC, distance correlation, MMD, Relief, ReliefF |
| 2 | Information-Theoretic Feature Selection | MI, CMI, mRMR, JMI, CMIM, DISR, ICAP, Rényi entropy |
| 3 | Wrapper Methods at Scale | SFS, SBS, SFFS, SBFS, Boruta, beam search, RFE |
| 4 | Embedded Methods — Regularisation & Tree Importance | Lasso, Elastic Net, Group Lasso, knockoffs, stability selection, SHAP |
| 5 | Genetic Algorithms for Feature Selection | Binary GA, operator design, fitness functions, DEAP |
| 6 | Advanced Evolutionary & Swarm Methods | PSO, DE, ACO, NSGA-II, NSGA-III, MOEA/D, CMA-ES |
| 7 | Feature Selection for Time Series — Temporal Dependencies | Granger causality, walk-forward, purged CV, spectral selection |
| 8 | Feature Selection for High-Dimensional & Wide Data | SIS, ISIS, compact GA, EDA, random projection, sparse PCA |
| 9 | Causal Feature Selection | PC algorithm, FCI, GES, ICP, double ML, IV regression |
| 10 | Ensemble & Hybrid Feature Selection | Stability aggregation, Pareto ranking, meta-selectors |
| 11 | Production Feature Selection Pipelines | Feature store, drift monitoring, MLflow, re-selection triggers |

### Module 0: The Feature Selection Landscape

The dimensionality curse quantified: why accuracy degrades, variance inflates, and distance metrics fail as $p$ grows. A unified taxonomy of filter, wrapper, embedded, evolutionary, and causal methods. The bias-variance tradeoff through the lens of subset size. Benchmark datasets used throughout the course. Pitfalls of naive selection: leakage, double dipping, selection bias in reported performance.

### Module 1: Statistical Filter Methods — Beyond Correlation

Pearson correlation and its failure modes with nonlinear relationships and heteroskedasticity. The F-test for relevance and its assumptions. Hilbert-Schmidt Independence Criterion (HSIC) as a kernel-based dependence measure with finite-sample bounds. Distance correlation: zero implies independence without distributional assumptions. Maximum Mean Discrepancy (MMD) for covariate shift detection. Relief and ReliefF: instance-based estimation of feature quality in multi-class settings. Minimum redundancy criteria via pairwise filter combinations.

### Module 2: Information-Theoretic Feature Selection

Entropy, mutual information, and conditional mutual information from measure-theoretic foundations. Estimation of MI from finite samples: histogram, KSG estimator, MINE neural estimator. Minimum Redundancy Maximum Relevance (mRMR): the greedy approximation and its theoretical gap from the optimal solution. Joint Mutual Information (JMI) and its complementarity-capturing property. Conditional Infomax Feature Extraction (CMIM). Double Input Symmetrical Relevance (DISR). Interaction Capping (ICAP). Rényi entropy generalisation. Transfer entropy for directed information flow. Practical comparison across UCI datasets.

### Module 3: Wrapper Methods at Scale

Sequential Forward Selection (SFS) and Sequential Backward Selection (SBS): complexity, early stopping, and nesting bias. Sequential Floating Forward/Backward Search (SFFS/SBFS): the floating step reduces the greedy trap. Boruta: all-relevant selection using shadow features and binomial testing — correct for multiple comparisons. Beam search: width-$k$ exploration of the lattice. Recursive Feature Elimination (RFE) with cross-validated stopping. Nested cross-validation architecture to prevent wrapper leakage. Computational cost management: parallelisation and surrogate model acceleration.

### Module 4: Embedded Methods — Regularisation & Tree Importance

Lasso geometry: the $\ell_1$ constraint induces sparsity at corners. Lars algorithm for the full regularisation path. Elastic Net for correlated feature groups. Group Lasso for structured sparsity. Stability selection: subsampling + Lasso to control per-feature false discovery rate. The model-X knockoff filter: exact FDR control under arbitrary dependence when the feature distribution is known. Debiased Lasso for valid post-selection inference. Tree-based importance: mean decrease impurity vs. permutation importance — bias sources and corrections. SHAP TreeExplainer for consistent, locally-accurate attribution.

### Module 5: Genetic Algorithms for Feature Selection

Binary chromosome encoding of feature subsets. Fitness function design: validation loss, AIC/BIC penalised objectives, multi-objective scalarisation. Selection operators: tournament, roulette wheel, rank-based. Crossover operators: single-point, two-point, uniform — theoretical mixing properties. Mutation rate scheduling. Elitism and generational vs. steady-state replacement. Bloat control and parsimony pressure. Implementation with DEAP: custom individuals, toolbox configuration, hall of fame. Convergence diagnostics and premature convergence detection. Benchmark against greedy wrappers.

### Module 6: Advanced Evolutionary & Swarm Methods

Particle Swarm Optimisation: velocity update, inertia weight, cognitive and social components, sigmoid binarisation for discrete spaces. Differential Evolution: mutation strategies (rand/1, best/2, current-to-best), crossover rate, population diversity maintenance. Ant Colony Optimisation: pheromone matrices for feature graphs, evaporation schedules. NSGA-II: non-dominated sorting, crowding distance, Pareto front approximation for simultaneous optimisation of accuracy and cardinality. NSGA-III: reference-point-based selection for many-objective problems. MOEA/D: decomposition into scalar subproblems with neighbourhood cooperation. CMA-ES with binarisation for continuous relaxation. Island model parallelism and migration policies. Surrogate-assisted evaluation for expensive fitness functions. Memetic algorithms: local search hybridisation.

### Module 7: Feature Selection for Time Series — Temporal Dependencies

Why i.i.d. selection methods fail on time series: autocorrelation inflates apparent relevance, lookahead leakage corrupts splits. Walk-forward cross-validation: expanding and sliding window variants. Purged cross-validation: removing training samples whose label overlaps the test period. Embargo gaps: quarantine samples adjacent to the test fold. Granger causality: VAR-based testing, lag selection, spurious causality with non-stationary series. Spectral feature selection: frequency-domain relevance via coherence and transfer functions. Regime-dependent selection: features whose relevance switches across hidden states. Feature drift monitoring in deployed time series models.

### Module 8: Feature Selection for High-Dimensional & Wide Data

The $p \gg n$ regime: ordinary Lasso breaks down, SIS provides a computationally feasible first screen. Sure Independence Screening (SIS): marginal correlation screening with sure-screening property. Iterative SIS (ISIS): iterative conditioning removes spurious survivors. Compact genetic algorithms and Estimation of Distribution Algorithms (EDA): probabilistic model of the promising region. Random projection for dimensionality reduction before selection. Sparse PCA: structured sparsity in loading vectors. Debiased Lasso for valid inference in high dimensions. Two-stage screening-then-selection pipelines. Benchmark on genomics and financial factor datasets with $p > 10{,}000$.

### Module 9: Causal Feature Selection

The distinction between predictive and causal relevance: a spurious feature can have high MI while a causal feature has low MI in observational data. The do-calculus: $P(Y \mid do(X))$ versus $P(Y \mid X)$. Markov blanket as the minimal causal sufficient set. Constraint-based discovery: PC algorithm and FCI for latent confounders. Score-based discovery: Greedy Equivalence Search (GES). Invariant Causal Prediction (ICP): features whose conditional distribution of $Y$ given $X_S$ is invariant across environments select the causal parents. Double ML: orthogonal estimation for causal feature coefficients under high-dimensional nuisance. Instrumental variable selection. Implementation with DoWhy and CausalLearn.

### Module 10: Ensemble & Hybrid Feature Selection

Aggregating multiple selectors reduces variance: majority voting, rank aggregation, and weighted combination. Stability analysis: Kuncheva's consistency index, Jaccard similarity across bootstrap runs. Pareto-front ranking across accuracy, cardinality, and stability objectives. Meta-selectors: train a second model on selector outputs. Sequential hybrid pipelines: filter to reduce search space, then wrapper or evolutionary search. Optuna for hyperparameter optimisation of the selector itself. Empirical comparison of ensemble strategies across 12 benchmark datasets.

### Module 11: Production Feature Selection Pipelines

Feature stores: centralised computation and serving, point-in-time correctness, preventing training-serving skew. MLflow experiment tracking for selection runs: parameter logging, metric comparison, artifact storage. Population Stability Index (PSI) for covariate drift detection between training and serving distributions. KS test for continuous feature drift. Automated re-selection triggers: drift thresholds, scheduled retraining, online monitoring dashboards. A/B testing framework for evaluating a new feature set against a deployed model. Deployment patterns: offline batch selection vs. online adaptive selection. Regulatory considerations: feature explainability requirements in financial and healthcare contexts.

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| `scikit-learn` | Base estimators, pipeline API, cross-validation, RFE |
| `numpy` / `pandas` | Numerical computation, data wrangling |
| `statsmodels` | Granger causality, VAR, OLS with inference |
| `DEAP` | Genetic algorithms, PSO, multi-objective evolutionary computation |
| `LightGBM` | Gradient-boosted trees for embedded selection and fitness evaluation |
| `XGBoost` | Alternative GBDT with SHAP integration |
| `scipy` | Statistical tests, distance metrics, sparse linear algebra |
| `DoWhy` | Causal modelling, do-calculus, IV estimation |
| `CausalLearn` | PC algorithm, FCI, GES discovery |
| `Optuna` | Hyperparameter optimisation for selector configuration |
| `MLflow` | Experiment tracking, model registry, artifact management |
| `matplotlib` / `seaborn` | Static visualisation |
| `plotly` | Interactive Pareto fronts, stability heatmaps |
| `pyarrow` / `feast` | Feature store integration |

**Python version:** 3.11+
**Installation:**
```bash
pip install scikit-learn numpy pandas statsmodels deap lightgbm xgboost scipy \
    dowhy causal-learn optuna mlflow matplotlib seaborn plotly pyarrow
```

---

## Quick-Starts

Quick-starts are entry-point notebooks that produce meaningful results in under 2 minutes. Each demonstrates a complete selection workflow end-to-end.

| Notebook | What You Get in 2 Minutes |
|----------|--------------------------|
| `00_feature_selection_in_5_lines.ipynb` | mRMR selection on a real tabular dataset, ranked feature list with MI scores |
| `01_boruta_quick_start.ipynb` | Boruta all-relevant selection with confirmed/tentative/rejected breakdown |
| `02_nsga2_quick_start.ipynb` | Multi-objective GA returning Pareto front of accuracy vs. feature count |
| `03_time_series_purged_cv.ipynb` | Walk-forward selection pipeline with embargo on financial returns data |
| `04_causal_markov_blanket.ipynb` | PC algorithm Markov blanket on synthetic DAG with ground truth comparison |
| `05_production_drift_check.ipynb` | PSI and KS drift report comparing training vs. production feature distributions |

---

## Templates

Production-ready Python scaffolds. Copy into your project and fill in the dataset and model.

| Template | Description |
|----------|-------------|
| `filter_pipeline_template.py` | Composable filter chain: variance, correlation, HSIC, MI with configurable thresholds |
| `boruta_template.py` | Boruta with shadow features, perc/alpha configuration, result serialisation |
| `ga_selector_template.py` | DEAP-based GA selector with pluggable fitness, checkpoint/resume, convergence plots |
| `nsga2_selector_template.py` | Multi-objective NSGA-II with Pareto front visualisation and solution selection |
| `pso_selector_template.py` | PSO with sigmoid binarisation, velocity clamping, and population diversity monitoring |
| `purged_cv_template.py` | Walk-forward CV with purging and embargo for any sklearn-compatible estimator |
| `stability_selection_template.py` | Subsampling Lasso with FDR control, stability scores, and selection threshold sweep |
| `knockoff_filter_template.py` | Model-X knockoffs with Gaussian and LASSO statistics, FDR-q target |
| `causal_icp_template.py` | ICP with multiple environment support and invariant set extraction |
| `mlflow_selection_tracker.py` | MLflow wrapper logging params, metrics, feature sets, and selector artifacts |
| `drift_monitor_template.py` | PSI and KS monitoring for deployed feature sets with alert thresholds |
| `production_pipeline_template.py` | End-to-end sklearn Pipeline from raw features through selection to model with MLflow |

---

## Recipes

Copy-paste patterns for the most common feature selection tasks.

| Recipe | Use Case |
|--------|---------|
| `mrmr_from_scratch.py` | mRMR greedy selection without external libraries, works on any DataFrame |
| `mutual_info_ksg.py` | KSG estimator for MI with continuous variables, bandwidth selection |
| `granger_causality_screen.py` | Pairwise Granger testing with lag selection and FDR correction |
| `shap_importance_ranking.py` | LightGBM SHAP values to feature ranking with stability bootstrap |
| `pareto_front_plot.py` | Matplotlib Pareto front from NSGA-II logbook, colour-coded by generation |
| `stability_heatmap.py` | Bootstrap selection frequency heatmap across methods |
| `nested_cv_wrapper.py` | Nested CV for unbiased wrapper evaluation with outer/inner split management |
| `transfer_entropy.py` | Transfer entropy estimation between two time series using KSG |
| `psi_report.py` | Population Stability Index report for all features between two DataFrames |
| `ensemble_vote.py` | Majority-vote ensemble of N selectors with configurable threshold |

---

## Projects

Portfolio projects applying course methods to real problems. All use publicly available datasets.

| Project | Dataset | Methods Applied |
|---------|---------|----------------|
| `project_01_genomics_screen.ipynb` | GSE high-dimensional gene expression | SIS, ISIS, sparse PCA, stability selection — $p = 20{,}000$, $n = 200$ |
| `project_02_financial_factors.ipynb` | Kenneth French factor library + Compustat | mRMR, Boruta, ICP for regime-robust equity signals |
| `project_03_energy_trading.ipynb` | EIA natural gas storage + weather + prices | Granger causality, purged CV, NSGA-II for multi-objective trading signal selection |
| `project_04_credit_risk.ipynb` | UCI credit default + HMDA mortgage data | Knockoff filter with FDR control for regulatory-compliant feature selection |
| `project_05_production_pipeline.ipynb` | NYC taxi demand (streaming simulation) | End-to-end: filter screen → evolutionary search → MLflow tracking → drift monitoring |

---

## Course Philosophy

This course follows the project-wide philosophy: working code first, theory contextually. Every concept is demonstrated with runnable code on real data before the mathematical machinery is introduced. Notebooks are 15 minutes maximum — if a topic needs more space, it gets multiple notebooks. There are no grading rubrics, no exams, and no synthetic data.

The goal is a portfolio of production-quality feature selection pipelines you can deploy on Monday morning, paired with enough theoretical depth that you can diagnose failures, adapt methods to novel settings, and read the research literature independently.

---

## Navigating the Course

**For the systematic learner:** Work through modules 0–11 in order. Each module's notebooks build on the previous.

**For the practitioner with a specific problem:**
- Tabular classification/regression with $p < 500$: Modules 1–4, then 10
- Time series with $n < 5{,}000$: Modules 7, then 3–4
- High-dimensional ($p > 1{,}000$): Module 8 first, then 5–6
- Causal / distribution-shift robustness: Module 9, then 10
- Already in production, need monitoring: Module 11 standalone

**For the researcher:** Module 2 (information theory), Module 6 (advanced evolutionary), and Module 9 (causal) contain the deepest theoretical content with research paper connections.

**Cross-module connections:** Every guide ends with a "Cross-Module Connections" section that explains explicitly how the current method relates to methods taught in preceding modules, and where the method is used in later modules. These sections are the connective tissue of the course — read them even when skipping modules.

---

## Repository Layout

```
advanced-feature-selection/
├── README.md
├── modules/
│   ├── module_00_landscape/
│   ├── module_01_statistical_filters/
│   ├── module_02_information_theory/
│   ├── module_03_wrappers/
│   ├── module_04_embedded_methods/
│   ├── module_05_genetic_algorithms/
│   ├── module_06_evolutionary_swarm/
│   ├── module_07_time_series/
│   ├── module_08_high_dimensional/
│   ├── module_09_causal/
│   ├── module_10_ensemble_hybrid/
│   └── module_11_production/
├── quick-starts/
├── templates/
├── recipes/
├── projects/
└── resources/
    ├── glossary.md
    ├── notation_guide.md
    └── bibliography.md
```

Each module contains `guides/` (concept guides and companion slide decks), `notebooks/` (15-minute Jupyter notebooks), `exercises/` (self-check Python exercises), and `resources/` (readings and figures).

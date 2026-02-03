# Capstone Project: Genetic Algorithm Feature Selection Pipeline

## Overview

Build a complete genetic algorithm-based feature selection system for a real-world machine learning problem. Implement custom GA operators, optimize for multiple objectives, and compare performance against traditional feature selection methods.

**Weight:** 30% of final grade
**Duration:** Weeks 7-9

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **GA Implementation:** Building genetic algorithms with custom operators
2. **Feature Selection:** Applying wrapper methods to reduce dimensionality
3. **Multi-Objective Optimization:** Balancing accuracy and parsimony
4. **Performance Evaluation:** Rigorous testing with cross-validation
5. **Comparison:** Benchmarking GA against filter and embedded methods
6. **Production Code:** Creating reusable, well-tested feature selection pipelines

---

## Project Requirements

Choose ONE application domain:

### Option A: High-Dimensional Regression

Feature selection for regression with many features.

**Example Datasets:**
- **Communities & Crime:** Predict crime rates (100+ features)
- **Bike Sharing:** Demand forecasting with temporal and weather features
- **Energy Efficiency:** Building performance prediction
- **Custom:** Your own regression dataset (p > 50 features)

**Requirements:**
- Continuous target variable
- At least 50 features
- Evaluate with RMSE or MAE

### Option B: Classification with Class Imbalance

Feature selection for imbalanced classification.

**Example Datasets:**
- **Credit Card Fraud:** Highly imbalanced (30 features)
- **Customer Churn:** Telecom or subscription service
- **Medical Diagnosis:** Disease prediction
- **Custom:** Your own imbalanced dataset

**Requirements:**
- Binary or multi-class classification
- Class imbalance (minority class < 20%)
- Evaluate with F1, AUC-ROC, or precision-recall

### Option C: Time Series Forecasting

Feature selection for time series with lagged features.

**Example Datasets:**
- **Stock Returns:** Predict using technical indicators and fundamentals
- **Energy Load:** Forecast using weather and calendar features
- **Sales Forecasting:** Product sales with promotions and seasonality
- **Custom:** Your own time series dataset

**Requirements:**
- Walk-forward validation
- Temporal features (lags, rolling statistics)
- At least 30 potential features

### Option D: Custom Proposal (Requires Approval)

Propose your own feature selection problem. Must include:
- Dataset with p ≥ 30 features
- Clear prediction task with evaluation metric
- Justification for GA approach
- Baseline methods for comparison

---

## Core Requirements (Must Complete All)

### 1. Data Preparation (10 points)

- [ ] Obtain dataset with sufficient size (n ≥ 500, p ≥ 30)
- [ ] Handle missing values appropriately
- [ ] Engineer additional features if appropriate
- [ ] Create train/validation/test splits (or CV folds)
- [ ] Document feature descriptions

**Grading:**
- Dataset quality and size: 4 points
- Preprocessing appropriateness: 3 points
- Documentation: 3 points

### 2. GA Implementation (30 points)

Implement genetic algorithm with:

**Required Components:**
- [ ] Binary chromosome encoding (feature selection vector)
- [ ] Fitness function combining accuracy and parsimony
- [ ] Population initialization strategy
- [ ] Selection operator (tournament, roulette, or rank)
- [ ] Crossover operator (single-point, two-point, or uniform)
- [ ] Mutation operator (bit-flip with adaptive rate)
- [ ] Elitism mechanism
- [ ] Convergence detection

**Implementation Options:**
- **Using DEAP:** Implement custom fitness and evaluation
- **From Scratch:** Full GA implementation
- **Hybrid:** DEAP framework with custom operators

**Grading:**
- GA correctness: 15 points
- Operator quality: 8 points
- Code organization: 5 points
- Documentation: 2 points

### 3. Fitness Function Design (15 points)

- [ ] Define multi-objective fitness: accuracy + parsimony
- [ ] Justify parsimony penalty weight
- [ ] Implement cross-validation for fitness evaluation
- [ ] Handle edge cases (empty feature set, all features)
- [ ] Optional: Implement Pareto-based multi-objective GA (NSGA-II)

**Example Fitness:**
```python
def fitness(chromosome):
    selected_features = [i for i, bit in enumerate(chromosome) if bit == 1]
    if len(selected_features) == 0:
        return 0,  # Minimum fitness

    X_selected = X[:, selected_features]
    cv_scores = cross_val_score(model, X_selected, y, cv=5)
    accuracy = cv_scores.mean()

    # Parsimony penalty
    n_features = len(selected_features)
    parsimony_penalty = lambda_param * (n_features / total_features)

    fitness = accuracy - parsimony_penalty
    return fitness,
```

**Grading:**
- Fitness design: 7 points
- CV implementation: 5 points
- Handling edge cases: 3 points

### 4. Hyperparameter Tuning (10 points)

Experiment with and justify:
- [ ] Population size (test at least 3 values)
- [ ] Number of generations
- [ ] Crossover probability
- [ ] Mutation probability
- [ ] Tournament size (if using tournament selection)
- [ ] Parsimony penalty weight (lambda)

**Requirements:**
- Systematic experimentation
- Plot convergence for different settings
- Justify final hyperparameters

**Grading:**
- Experimentation thoroughness: 5 points
- Justification quality: 3 points
- Visualization: 2 points

### 5. Comparison with Baselines (20 points)

Compare GA against at least THREE methods:

**Filter Methods:**
- [ ] Correlation-based (for regression)
- [ ] Mutual information
- [ ] Chi-square (for classification)
- [ ] Variance threshold

**Wrapper Methods:**
- [ ] Recursive Feature Elimination (RFE)
- [ ] Forward/Backward Selection

**Embedded Methods:**
- [ ] L1 Regularization (Lasso)
- [ ] Tree-based importance (Random Forest, XGBoost)

**Evaluation:**
- Compare on same train/test splits
- Report accuracy and number of features
- Compare computation time
- Statistical significance testing (optional)

**Grading:**
- Method diversity: 8 points
- Fair comparison: 7 points
- Analysis quality: 5 points

### 6. Analysis & Interpretation (10 points)

- [ ] Analyze selected features (which are chosen?)
- [ ] Feature stability across GA runs
- [ ] Convergence analysis (fitness over generations)
- [ ] Diversity metrics (population diversity)
- [ ] Pareto front visualization (if multi-objective)

**Grading:**
- Feature analysis: 4 points
- Convergence analysis: 3 points
- Interpretation quality: 3 points

### 7. Code Quality & Testing (5 points)

- [ ] Modular, reusable code
- [ ] Unit tests for key functions
- [ ] Docstrings and comments
- [ ] Reproducible (random seeds set)
- [ ] Requirements.txt or environment.yml

**Grading:**
- Code quality: 3 points
- Testing: 1 point
- Documentation: 1 point

---

## Extension Options (Choose 1-2 for Bonus)

Each extension worth up to 5 bonus points:

1. **Multi-Objective NSGA-II**
   - Implement Pareto-based selection
   - Generate Pareto front of solutions
   - Knee point selection method

2. **Adaptive Operators**
   - Mutation rate that adapts to diversity
   - Self-adaptive crossover
   - Dynamic population sizing

3. **Hybrid GA**
   - Combine with local search (memetic algorithm)
   - Use filter methods for initialization
   - Incorporate domain knowledge constraints

4. **Feature Engineering Integration**
   - Automatically create interaction features
   - Polynomial features
   - GA selects from expanded feature space

5. **Distributed/Parallel GA**
   - Island model GA
   - Parallel fitness evaluation
   - Speed comparison vs. serial

6. **Comprehensive Benchmarking**
   - Multiple datasets
   - Statistical comparison (Wilcoxon, Friedman test)
   - Critical difference diagrams

---

## Milestones & Checkpoints

### Milestone 1: Data & Basic GA (Week 7) — 10%
**Deliverable:** Jupyter notebook + dataset

- Dataset prepared and split
- Basic GA implemented and running
- Initial feature selection results

**Grading:**
- Data preparation: 4 points
- GA functionality: 5 points
- Documentation: 1 point

### Milestone 2: Optimization & Baselines (Week 8) — 15%
**Deliverable:** Code + results

- Hyperparameter tuning complete
- Baseline methods implemented
- Preliminary comparison

**Grading:**
- Tuning quality: 7 points
- Baselines implemented: 6 points
- Comparison: 2 points

### Milestone 3: Final Submission (Week 9) — 75%
**Deliverables:** Complete pipeline + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (4-6 pages, excluding appendices)

1. **Introduction** (0.5-1 page)
   - Problem description and motivation
   - Why feature selection matters for this problem
   - Why GA is appropriate
   - Preview of findings

2. **Data & Preprocessing** (0.5-1 page)
   - Dataset description
   - Feature types and descriptions
   - Preprocessing steps
   - Train/test split strategy

3. **Methodology** (1.5-2 pages)
   - GA design (encoding, operators, fitness)
   - Hyperparameter settings and tuning
   - Baseline methods
   - Evaluation metrics

4. **Results** (1.5-2 pages)
   - GA convergence plots
   - Selected features analysis
   - Comparison table (GA vs. baselines)
   - Performance on test set

5. **Discussion** (0.5-1 page)
   - Interpretation of selected features
   - GA strengths and limitations for this problem
   - Computational efficiency considerations
   - Recommendations

6. **Appendix**
   - Hyperparameter tuning results
   - Full feature list with selection frequency
   - Additional plots (diversity, Pareto front)

---

## Presentation Rubric

### Structure (10 minutes total)
- Problem and motivation: 1-2 min
- GA approach: 2-3 min
- Results and comparison: 3-4 min
- Key insights: 1-2 min
- Q&A: time remaining

### Evaluation Criteria

| Criterion | Excellent (5) | Good (4) | Adequate (3) | Needs Work (1-2) |
|-----------|---------------|----------|--------------|------------------|
| **Clarity** | Crystal clear, engaging | Clear, well-paced | Understandable | Confusing |
| **Technical Depth** | Deep GA understanding | Solid grasp | Basic understanding | Superficial |
| **Results** | Compelling, well-analyzed | Good results | Adequate results | Weak or incomplete |
| **Comparison** | Fair, insightful | Good comparison | Basic comparison | Poor or missing |
| **Q&A** | Excellent responses | Answers well | Struggles with some | Cannot defend |

---

## Final Grading Rubric

### Data Preparation (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | High-quality dataset; excellent preprocessing; thorough documentation |
| 7-8 | Good dataset; solid preprocessing; adequate documentation |
| 5-6 | Adequate dataset; basic preprocessing; minimal documentation |
| 0-4 | Poor dataset; weak preprocessing; no documentation |

### GA Implementation (30 points)
| Points | Criteria |
|--------|----------|
| 27-30 | Excellent implementation; all operators correct; well-organized code |
| 21-26 | Good implementation; mostly correct; reasonably organized |
| 15-20 | Adequate implementation; some issues; poorly organized |
| 0-14 | Incomplete or incorrect implementation |

### Fitness Function (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Excellent design; robust CV; handles edge cases |
| 10-12 | Good design; solid CV; handles most cases |
| 7-9 | Basic design; adequate CV; some edge cases missed |
| 0-6 | Poor design; weak CV; many edge cases missed |

### Hyperparameter Tuning (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Systematic tuning; thorough experimentation; well-justified |
| 7-8 | Good tuning; solid experimentation; reasonably justified |
| 5-6 | Basic tuning; limited experimentation; weak justification |
| 0-4 | Minimal or no tuning; poor justification |

### Baseline Comparison (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Diverse baselines; fair comparison; excellent analysis |
| 14-17 | Good baselines; solid comparison; good analysis |
| 10-13 | Basic baselines; adequate comparison; limited analysis |
| 0-9 | Few baselines; unfair comparison; weak analysis |

### Analysis & Interpretation (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Deep analysis; insightful interpretation; thorough |
| 7-8 | Good analysis; solid interpretation; adequate |
| 5-6 | Basic analysis; limited interpretation; incomplete |
| 0-4 | Minimal analysis; weak interpretation; missing |

### Code Quality (5 points)
| Points | Criteria |
|--------|----------|
| 5 | Excellent code; well-tested; professional documentation |
| 4 | Good code; some tests; solid documentation |
| 3 | Adequate code; minimal tests; basic documentation |
| 0-2 | Poor code; no tests; no documentation |

---

## Technical Specifications

### Minimum Requirements
- **Python 3.8+**
- **Dataset:** n ≥ 500 samples, p ≥ 30 features
- **GA Generations:** Minimum 50 (or until convergence)
- **Population Size:** Minimum 50
- **Cross-Validation:** 5-fold minimum for fitness evaluation
- **Baseline Methods:** At least 3

### Recommended Libraries

```python
# GA Framework
from deap import base, creator, tools, algorithms

# ML Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC, SVR

# Feature Selection
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, mutual_info_classif,
    RFE, SelectFromModel
)

# Evaluation
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# Utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

### Sample GA Structure (DEAP)

```python
import random
from deap import base, creator, tools, algorithms

# Define fitness (maximize)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA
population = toolbox.population(n=100)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population, log = algorithms.eaSimple(population, toolbox,
                                       cxpb=0.7, mutpb=0.2,
                                       ngen=50, stats=stats,
                                       halloffame=hof, verbose=True)

# Best solution
best_individual = hof[0]
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
```

---

## Academic Integrity

- This is individual work
- You may discuss concepts but code must be your own
- Cite any external code or algorithms used
- Document any AI assistance for coding
- Understand and be able to explain all implementations

---

## Resources

### Datasets
- **UCI ML Repository:** [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/)
- **Kaggle:** [kaggle.com/datasets](https://www.kaggle.com/datasets)
- **OpenML:** [openml.org](https://www.openml.org)
- **Scikit-learn datasets:** `sklearn.datasets`

### Documentation
- **DEAP:** [deap.readthedocs.io](https://deap.readthedocs.io)
- **Scikit-learn Feature Selection:** [scikit-learn.org/stable/modules/feature_selection](https://scikit-learn.org/stable/modules/feature_selection.html)

### Papers
- Xue, B., et al. (2016). "A Survey on Evolutionary Computation Approaches to Feature Selection." IEEE TEVC.
- Deb, K., et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE TEVC.

---

## Submission Instructions

1. **Create repository:**
   ```
   ga-feature-selection/
   ├── README.md
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── src/
   │   ├── ga.py                # GA implementation
   │   ├── fitness.py           # Fitness functions
   │   ├── baselines.py         # Baseline methods
   │   └── utils.py             # Helper functions
   ├── tests/
   │   └── test_ga.py           # Unit tests
   ├── notebooks/
   │   ├── 01_data_prep.ipynb
   │   ├── 02_ga_experiments.ipynb
   │   └── 03_comparison.ipynb
   ├── results/
   │   ├── figures/
   │   └── metrics.csv
   ├── docs/
   │   └── report.pdf
   └── requirements.txt
   ```

2. **Submit via course platform:**
   - GitHub repository link
   - Technical report (PDF)
   - Presentation slides (PDF)

3. **Reproducibility:**
   - Random seeds set throughout
   - Requirements file with versions
   - Clear README with run instructions
   - Sample data or download instructions

---

*"Feature selection is not just dimension reduction—it's about understanding what matters. GAs explore feature space more thoroughly than greedy methods, often finding better subsets."*

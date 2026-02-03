# Capstone Project: Panel Data Analysis for Policy or Business Application

## Overview

Conduct a rigorous panel data analysis to answer an important empirical question in economics, business, or policy. Apply fixed effects, random effects, and diagnostic tests to control for unobserved heterogeneity and produce credible causal or predictive insights.

**Weight:** 30% of final grade
**Duration:** Weeks 8-10

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **Panel Data Structure:** Organizing and exploring longitudinal data
2. **Model Selection:** Choosing between pooled OLS, fixed effects, and random effects
3. **Estimation:** Implementing panel estimators with appropriate standard errors
4. **Inference:** Conducting hypothesis tests and interpreting results
5. **Diagnostics:** Testing assumptions and assessing model adequacy
6. **Communication:** Presenting econometric results to technical and non-technical audiences

---

## Project Options

Choose ONE empirical question:

### Option A: Policy Evaluation

Evaluate the impact of a policy intervention using panel data.

**Examples:**
- Effect of minimum wage changes on employment (state-level panel)
- Impact of carbon taxes on emissions (country-level panel)
- Effect of education policies on student outcomes (school district panel)
- Impact of corporate tax rates on investment (firm-level panel)

**Requirements:**
- Clear treatment/control variation
- Panel data spanning policy change
- Justification of identifying assumptions

### Option B: Determinants of Performance

Analyze factors driving performance variation across entities.

**Examples:**
- Determinants of firm profitability (financial panel)
- Drivers of economic growth (country-level panel)
- Factors affecting stock returns (equity panel)
- Predictors of hospital quality (hospital panel)

**Requirements:**
- Multiple potential explanatory variables
- Economic theory motivating analysis
- Interpretation of within vs. between variation

### Option C: Panel Prediction Model

Build a predictive model using panel structure.

**Examples:**
- Forecasting commodity prices (futures panel)
- Predicting firm bankruptcy (corporate panel)
- Credit default prediction (loan-level panel)
- Sales forecasting (product-store-week panel)

**Requirements:**
- Clear prediction task with evaluation metric
- Out-of-sample validation
- Comparison with time-series or cross-sectional baselines

### Option D: Custom Proposal (Requires Approval)

Propose your own panel analysis. Must include:
- Panel dataset with N ≥ 50 entities, T ≥ 5 periods
- Clear research question
- Justification for panel methods
- Appropriate estimation strategy

---

## Core Requirements (Must Complete All)

### 1. Data Acquisition & Preparation (15 points)

- [ ] Obtain panel dataset (minimum: N=50, T=5)
- [ ] Convert to long format if necessary
- [ ] Handle missing data appropriately
- [ ] Create time-varying and time-invariant variables
- [ ] Document data sources and cleaning steps

**Data Sources:**
- **Economic:** FRED, World Bank, Penn World Tables
- **Financial:** Compustat, CRSP, Bloomberg
- **Policy:** Census, BLS, state/local government data
- **Custom:** Industry data, proprietary sources

**Grading:**
- Data quality and size: 6 points
- Variable construction: 5 points
- Documentation: 4 points

### 2. Exploratory Data Analysis (10 points)

- [ ] Verify panel structure (balanced vs. unbalanced)
- [ ] Descriptive statistics (overall, between, within)
- [ ] Visualization by entity and time
- [ ] Identify time-invariant vs. time-varying regressors
- [ ] Preliminary evidence on research question

**Grading:**
- Panel structure analysis: 4 points
- Visualization quality: 3 points
- Preliminary insights: 3 points

### 3. Model Estimation (30 points)

Estimate at least THREE specifications:

**Required:**
- [ ] Pooled OLS (baseline)
- [ ] Fixed Effects (entity)
- [ ] Random Effects

**Choose ONE additional:**
- [ ] Two-way fixed effects (entity + time)
- [ ] First-difference estimator
- [ ] Between estimator
- [ ] Dynamic panel (Arellano-Bond) if applicable

For each model:
- [ ] Use appropriate standard errors (clustered minimum)
- [ ] Justify specification
- [ ] Present results clearly

**Grading:**
- Model correctness: 12 points
- Standard error handling: 8 points
- Specification justification: 6 points
- Presentation: 4 points

### 4. Model Selection & Diagnostics (20 points)

- [ ] F-test for fixed effects
- [ ] Hausman test (FE vs. RE)
- [ ] Test for autocorrelation (Wooldridge test)
- [ ] Test for heteroskedasticity
- [ ] Check for multicollinearity (VIF)
- [ ] Residual analysis
- [ ] Justify final model selection

**Grading:**
- Statistical tests: 10 points
- Diagnostic quality: 6 points
- Model selection justification: 4 points

### 5. Interpretation & Economic Analysis (15 points)

- [ ] Interpret coefficients economically
- [ ] Distinguish within vs. between effects (if relevant)
- [ ] Discuss magnitude and significance
- [ ] Address causality vs. correlation
- [ ] Relate findings to research question
- [ ] Acknowledge limitations

**Grading:**
- Economic interpretation: 7 points
- Causal reasoning: 5 points
- Limitation discussion: 3 points

### 6. Robustness Checks (5 points)

Perform at least TWO:
- [ ] Alternative standard error specifications
- [ ] Different time periods or subsamples
- [ ] Alternative control variables
- [ ] Different dependent variable specification (e.g., logs)
- [ ] Addressing outliers or influential observations

**Grading:**
- Check appropriateness: 3 points
- Analysis quality: 2 points

### 7. Documentation & Communication (5 points)

- [ ] Well-organized code (Python or R)
- [ ] Clear README with reproduction steps
- [ ] Technical report (see template below)
- [ ] Presentation (see rubric below)

**Grading:**
- Code quality: 2 points
- Report quality: 2 points
- Presentation: 1 point

---

## Extension Options (Choose 1 for Bonus)

Each extension worth up to 5 bonus points:

1. **Instrumental Variables**
   - Address endogeneity with panel IV
   - Justify instrument validity
   - First-stage and reduced-form results

2. **Treatment Effects Analysis**
   - Implement difference-in-differences
   - Parallel trends test
   - Event study specification

3. **Nonlinear Panel Models**
   - Logit/probit for binary outcomes
   - Tobit for censored data
   - Count models (Poisson/NB)

4. **Dynamic Panel with GMM**
   - Arellano-Bond or Blundell-Bond
   - Discuss lagged dependent variable
   - Hansen J-test for overidentification

5. **Comprehensive Sensitivity Analysis**
   - Systematic robustness checks
   - Specification curve analysis
   - Bootstrap standard errors

---

## Milestones & Checkpoints

### Milestone 1: Data & EDA (Week 8) — 10%
**Deliverable:** Jupyter notebook + dataset

- Panel dataset loaded and structured
- EDA complete with visualizations
- Research question refined

**Grading:**
- Data preparation: 5 points
- EDA quality: 4 points
- Documentation: 1 point

### Milestone 2: Initial Models (Week 9) — 15%
**Deliverable:** Code + preliminary results

- Pooled OLS, FE, RE estimated
- Initial diagnostics
- Preliminary interpretation

**Grading:**
- Models estimated correctly: 8 points
- Diagnostics present: 5 points
- Code organization: 2 points

### Milestone 3: Final Submission (Week 10) — 75%
**Deliverables:** Complete analysis + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (5-7 pages, excluding appendices)

1. **Introduction** (1 page)
   - Research question and motivation
   - Why panel methods are appropriate
   - Preview of findings

2. **Data** (1-1.5 pages)
   - Description of dataset
   - Panel structure (N, T, balanced/unbalanced)
   - Variable definitions
   - Descriptive statistics table
   - Data sources

3. **Empirical Strategy** (1.5-2 pages)
   - Econometric model specification
   - Estimators used and why
   - Identifying assumptions (if causal)
   - Standard error specification

4. **Results** (1.5-2 pages)
   - Main regression table
   - Model selection tests (F-test, Hausman)
   - Preferred specification interpretation
   - Robustness checks

5. **Discussion** (0.5-1 page)
   - Economic/policy implications
   - Limitations
   - Comparison to related literature (if applicable)
   - Future research

6. **Appendix**
   - Additional tables (alternative specifications)
   - Diagnostic plots
   - Robustness check results

### Tables & Figures

**Required:**
- Table 1: Descriptive statistics (overall, between, within)
- Table 2: Main regression results (Pooled OLS, FE, RE)
- Table 3: Diagnostic tests summary
- Figure 1: Outcome trends by entity (sample)
- Figure 2: Residual diagnostics

---

## Presentation Rubric

### Structure (10 minutes total)
- Motivation and research question: 2 min
- Data and methodology: 2 min
- Results: 4 min
- Interpretation and implications: 1-2 min
- Q&A: time remaining

### Evaluation Criteria

| Criterion | Excellent (4) | Good (3) | Adequate (2) | Needs Work (1) |
|-----------|--------------|----------|--------------|----------------|
| **Clarity** | Crystal clear, engaging | Clear, well-paced | Understandable | Confusing |
| **Technical Rigor** | Correct methods, thoughtful | Solid approach, minor issues | Basic correctness | Errors or gaps |
| **Results Presentation** | Clear tables, insightful | Good tables, useful | Adequate | Unclear or incomplete |
| **Economic Insight** | Deep understanding, implications | Good interpretation | Basic interpretation | Weak or missing |
| **Q&A** | Handles well, defends choices | Answers most questions | Struggles with some | Cannot defend |

---

## Final Grading Rubric

### Data & Preparation (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Large, high-quality panel; excellent variables; thorough documentation |
| 10-12 | Good panel; solid variables; adequate documentation |
| 7-9 | Adequate panel; basic variables; minimal documentation |
| 0-6 | Small or poor-quality panel; weak variables |

### EDA (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Comprehensive EDA; excellent visualizations; insightful |
| 7-8 | Good EDA; solid visualizations; useful |
| 5-6 | Basic EDA; adequate visualizations; limited insights |
| 0-4 | Minimal EDA; poor visualizations |

### Model Estimation (30 points)
| Points | Criteria |
|--------|----------|
| 27-30 | Multiple correct models; appropriate SE; excellent justification; clear presentation |
| 21-26 | Good models; correct SE; solid justification; good presentation |
| 15-20 | Adequate models; some SE issues; weak justification; unclear presentation |
| 0-14 | Incorrect models; wrong SE; poor justification |

### Diagnostics & Selection (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Comprehensive tests; correct interpretation; well-justified selection |
| 14-17 | Good tests; solid interpretation; reasonable selection |
| 10-13 | Basic tests; adequate interpretation; weak selection |
| 0-9 | Minimal tests; poor interpretation; unjustified selection |

### Interpretation (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Deep economic understanding; causal reasoning; acknowledges limitations |
| 10-12 | Good interpretation; reasonable reasoning; some limitations noted |
| 7-9 | Basic interpretation; limited reasoning; few limitations |
| 0-6 | Weak interpretation; no reasoning; no limitations |

### Robustness (5 points)
| Points | Criteria |
|--------|----------|
| 5 | Multiple appropriate checks; thorough analysis |
| 4 | Good checks; solid analysis |
| 3 | Basic checks; adequate analysis |
| 0-2 | Minimal or inappropriate checks |

### Documentation (5 points)
| Points | Criteria |
|--------|----------|
| 5 | Excellent code; professional report; strong presentation |
| 4 | Good code; solid report; good presentation |
| 3 | Adequate code; acceptable report; basic presentation |
| 0-2 | Poor code; weak report; poor presentation |

---

## Technical Requirements

### Minimum Panel Size
- **Entities (N):** 50+
- **Time Periods (T):** 5+
- **Total Observations:** 250+ (balanced) or adequate coverage (unbalanced)

### Software

**Python (linearmodels):**
```python
from linearmodels import PanelOLS, RandomEffects
import pandas as pd

# Set panel index
df = df.set_index(['entity', 'time'])

# Fixed Effects
fe = PanelOLS(df['y'], df[['x1', 'x2']], entity_effects=True)
fe_res = fe.fit(cov_type='clustered', cluster_entity=True)

# Random Effects
re = RandomEffects(df['y'], df[['x1', 'x2']])
re_res = re.fit()

# Hausman test
from linearmodels.panel import compare
compare({'FE': fe_res, 'RE': re_res})
```

**R (plm):**
```r
library(plm)

# Fixed Effects
fe_model <- plm(y ~ x1 + x2, data=df, index=c("entity","time"),
                model="within")
summary(fe_model, vcov=vcovHC)

# Random Effects
re_model <- plm(y ~ x1 + x2, data=df, index=c("entity","time"),
                model="random")

# Hausman Test
phtest(fe_model, re_model)
```

---

## Academic Integrity

- This is individual work
- Cite all data sources
- You may discuss concepts but code must be your own
- Document any AI assistance for coding
- Understand and be able to explain all analyses

---

## Resources

### Textbooks & References
- Wooldridge, J. (2010). *Econometric Analysis of Panel Data*
- Stock & Watson (2020). *Introduction to Econometrics*
- Baltagi, B. (2021). *Econometric Analysis of Panel Data*

### Data Sources
- **FRED:** [fred.stlouisfed.org](https://fred.stlouisfed.org)
- **World Bank:** [data.worldbank.org](https://data.worldbank.org)
- **Penn World Tables:** [pwt.sas.upenn.edu](https://www.rug.nl/ggdc/productivity/pwt/)
- **Compustat / CRSP:** (via university access)
- **IPUMS:** [ipums.org](https://ipums.org)

### Software Documentation
- **linearmodels (Python):** [bashtage.github.io/linearmodels](https://bashtage.github.io/linearmodels/)
- **plm (R):** [cran.r-project.org/web/packages/plm](https://cran.r-project.org/web/packages/plm/)

---

## Submission Instructions

1. **Create repository:**
   ```
   panel-regression-capstone/
   ├── README.md
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── code/
   │   ├── 01_data_prep.py (or .R)
   │   ├── 02_eda.py
   │   ├── 03_models.py
   │   └── 04_diagnostics.py
   ├── results/
   │   ├── tables/
   │   └── figures/
   ├── docs/
   │   └── report.pdf
   └── requirements.txt (or environment.yml)
   ```

2. **Submit via course platform:**
   - GitHub repository link
   - Technical report (PDF)
   - Presentation slides (PDF)

3. **Reproducibility:**
   - Clear README with setup instructions
   - Environment file with package versions
   - Data download instructions or sample data
   - Code runs without errors

---

*"Panel data's power comes from controlling for what you cannot observe. Use this power wisely, test assumptions carefully, and interpret results humbly."*

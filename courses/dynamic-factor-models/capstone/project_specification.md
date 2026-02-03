# Capstone Project: Real-Time Macroeconomic Nowcasting System

## Project Overview

Build an end-to-end nowcasting system that produces real-time estimates of a macroeconomic indicator (GDP, industrial production, or employment) using dynamic factor models. Your system must handle mixed-frequency data, publication lags, and ragged-edge datasets—the real challenges of practical nowcasting.

**Duration:** 3-4 weeks
**Deliverables:** Code repository, technical report, presentation
**Weight:** 30% of course grade

---

## Learning Objectives

This project demonstrates mastery of:

1. **Data Pipeline Construction** - Acquiring, cleaning, and transforming real-time economic data
2. **Factor Model Estimation** - Implementing DFM estimation for large panels
3. **Nowcasting Methodology** - Handling mixed frequencies and publication lags
4. **Forecast Evaluation** - Using proper real-time vintage data for assessment
5. **Production Code Quality** - Writing reproducible, documented, testable code

---

## Project Requirements

### Core Requirements (Must Complete)

#### 1. Data Pipeline
- [ ] Acquire FRED-MD monthly dataset (or subset of 50+ series)
- [ ] Handle missing values and ragged edges appropriately
- [ ] Implement data transformations (stationarity, standardization)
- [ ] Create real-time data vintages for evaluation

#### 2. Factor Model
- [ ] Estimate dynamic factor model with at least 3 factors
- [ ] Use either PCA-based or state-space estimation
- [ ] Determine number of factors using information criteria
- [ ] Extract and interpret factor estimates

#### 3. Nowcasting Model
- [ ] Target variable: Quarterly GDP growth OR monthly IP growth
- [ ] Handle mixed frequencies (monthly indicators → quarterly target)
- [ ] Update nowcasts as new data arrives (simulate real-time)
- [ ] Produce point forecasts and uncertainty intervals

#### 4. Evaluation
- [ ] Pseudo out-of-sample evaluation (rolling window)
- [ ] Compare to benchmark (AR, random walk)
- [ ] Use RMSFE, MAE, and directional accuracy
- [ ] Analyze forecast errors by data availability

#### 5. Documentation
- [ ] README with setup instructions
- [ ] Code comments and docstrings
- [ ] Technical report (10-15 pages)
- [ ] Reproducibility (requirements.txt, random seeds)

### Extension Options (Choose 1-2)

#### Option A: Bayesian Estimation
- Implement Bayesian DFM using PyMC or NumPyro
- Compare posterior uncertainty to frequentist standard errors
- Use informative priors based on economic theory

#### Option B: Sparse Factor Model
- Implement targeted predictors or LASSO-based selection
- Compare dense vs sparse factor forecasts
- Analyze which predictors are selected

#### Option C: Real-Time Vintage Evaluation
- Use ALFRED for true real-time GDP vintages
- Evaluate forecast accuracy accounting for data revisions
- Decompose errors into timing vs revision components

#### Option D: Multiple Targets
- Nowcast multiple targets (GDP, IP, employment)
- Analyze factor relevance across targets
- Build combined "economic conditions" indicator

#### Option E: Visualization Dashboard
- Create interactive Streamlit/Dash application
- Real-time factor visualization
- Nowcast display with confidence bands

---

## Milestone Schedule

### Milestone 1: Data Pipeline (Week 1)
**Deliverable:** Working data acquisition and preprocessing pipeline

Checklist:
- [ ] FRED API access configured
- [ ] Data download script functional
- [ ] Transformation codes applied correctly
- [ ] Missing value handling implemented
- [ ] Ragged-edge structure preserved
- [ ] Unit tests for data functions

**Submission:** Code + brief write-up of data decisions

### Milestone 2: Factor Model (Week 2)
**Deliverable:** Estimated factor model with interpreted factors

Checklist:
- [ ] Factor extraction implemented
- [ ] Number of factors selected with justification
- [ ] Factor loadings analyzed and interpreted
- [ ] Factors visualized and compared to known series
- [ ] In-sample fit evaluated

**Submission:** Code + factor interpretation analysis

### Milestone 3: Nowcasting System (Week 3)
**Deliverable:** Working nowcasting pipeline with forecasts

Checklist:
- [ ] Target variable defined and linked to factors
- [ ] Mixed-frequency handling implemented
- [ ] Nowcast updating mechanism working
- [ ] Forecast uncertainty quantified
- [ ] Out-of-sample evaluation framework ready

**Submission:** Code + preliminary results

### Milestone 4: Final Submission (Week 4)
**Deliverable:** Complete project package

Checklist:
- [ ] All code cleaned and documented
- [ ] Technical report complete
- [ ] Evaluation results finalized
- [ ] Extension option implemented
- [ ] Presentation prepared

---

## Technical Report Structure

### 1. Introduction (1-2 pages)
- Problem motivation: Why nowcasting matters
- Target variable and data description
- Brief methodology overview

### 2. Data (2-3 pages)
- Data sources and acquisition
- Transformations applied (with justification)
- Handling of missing data and publication lags
- Descriptive statistics

### 3. Methodology (3-4 pages)
- Factor model specification
- Estimation approach (PCA, MLE, Bayesian)
- Nowcasting model (bridge equation, state-space)
- Number of factors selection

### 4. Results (3-4 pages)
- Factor interpretation (loadings analysis)
- In-sample fit
- Out-of-sample forecast accuracy
- Comparison to benchmarks
- Information gain analysis

### 5. Discussion (1-2 pages)
- Key findings
- Limitations
- Potential improvements
- Practical implications

### 6. Conclusion (0.5-1 page)
- Summary of contributions
- Recommendations for practitioners

### Appendix
- Additional tables and figures
- Code organization description
- Reproducibility instructions

---

## Evaluation Rubric

### Technical Implementation (40 points)

| Criterion | Excellent (36-40) | Good (28-35) | Adequate (20-27) | Needs Work (<20) |
|-----------|-------------------|--------------|------------------|------------------|
| Data Pipeline | Robust, handles edge cases, well-tested | Works correctly, minor issues | Functional but fragile | Broken or incomplete |
| Factor Model | Correct implementation, proper identification | Works with minor issues | Conceptual gaps | Major errors |
| Nowcasting | Proper mixed-freq handling, uncertainty | Works but limited uncertainty | Incomplete implementation | Not functional |
| Evaluation | Proper real-time methodology | Minor methodological issues | Significant issues | Incorrect methodology |

### Code Quality (20 points)

| Criterion | Excellent (18-20) | Good (14-17) | Adequate (10-13) | Needs Work (<10) |
|-----------|-------------------|--------------|------------------|------------------|
| Readability | Clean, consistent style, clear naming | Mostly readable | Inconsistent | Hard to follow |
| Documentation | Comprehensive docstrings, README | Adequate docs | Sparse documentation | Missing docs |
| Reproducibility | One-command execution, seeded | Minor manual steps | Significant setup needed | Not reproducible |
| Testing | Unit tests, validation checks | Some tests | Minimal testing | No tests |

### Analysis & Interpretation (25 points)

| Criterion | Excellent (23-25) | Good (18-22) | Adequate (13-17) | Needs Work (<13) |
|-----------|-------------------|--------------|------------------|------------------|
| Factor Interpretation | Insightful economic interpretation | Reasonable interpretation | Surface-level | Missing or incorrect |
| Results Analysis | Deep understanding, proper comparisons | Good analysis | Basic analysis | Weak analysis |
| Critical Thinking | Discusses limitations, alternatives | Some critical discussion | Limited critique | No critique |

### Report & Presentation (15 points)

| Criterion | Excellent (14-15) | Good (11-13) | Adequate (8-10) | Needs Work (<8) |
|-----------|-------------------|--------------|------------------|------------------|
| Writing Quality | Clear, concise, well-organized | Good writing, minor issues | Understandable but rough | Poor writing |
| Presentation | Engaging, clear visuals, good pacing | Good presentation | Adequate | Poor delivery |
| Completeness | All sections thorough | Minor gaps | Significant gaps | Incomplete |

---

## Data Sources

### Primary: FRED-MD
- Monthly macroeconomic database (127 series)
- Download: [FRED-MD](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
- Includes transformation codes

### Target Variable Options

**Quarterly GDP:**
- Series: GDPC1 (Real GDP)
- Source: BEA via FRED
- Release lag: ~1 month after quarter end

**Monthly Industrial Production:**
- Series: INDPRO
- Source: Federal Reserve
- Release lag: ~2 weeks

### Real-Time Vintages (Extension)
- ALFRED: [alfred.stlouisfed.org](https://alfred.stlouisfed.org)
- Historical vintages for real-time evaluation

---

## Starter Code

A skeleton repository is provided with:

```
capstone/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── data/
│   │   ├── download.py      # FRED data acquisition
│   │   ├── transform.py     # Transformations
│   │   └── ragged_edge.py   # Missing data handling
│   ├── models/
│   │   ├── factor_model.py  # DFM estimation
│   │   └── nowcast.py       # Nowcasting model
│   ├── evaluation/
│   │   ├── metrics.py       # Forecast metrics
│   │   └── vintage.py       # Real-time evaluation
│   └── utils/
│       └── plotting.py      # Visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_analysis.ipynb
│   └── 03_nowcasting_evaluation.ipynb
└── tests/
    ├── test_data.py
    └── test_models.py
```

---

## Submission Instructions

### Milestone Submissions
- Push to your course repository branch
- Tag milestone: `milestone-1`, `milestone-2`, etc.
- Submit link via course portal

### Final Submission
- Complete repository with all code
- PDF of technical report
- PDF/link to presentation slides
- Tag: `final-submission`

### Presentation
- 15 minutes + 5 minutes Q&A
- Focus on: methodology, key results, insights
- Live demo encouraged but not required

---

## Academic Integrity

- All code must be your own
- Cite any external resources or code snippets
- Collaboration on concepts is encouraged; code sharing is not
- AI assistants may be used for debugging; cite if used for significant code generation

---

## Tips for Success

1. **Start Early** - Data pipelines always take longer than expected
2. **Version Control** - Commit frequently with meaningful messages
3. **Test Incrementally** - Verify each component before moving on
4. **Document As You Go** - Don't leave documentation for the end
5. **Focus on Core Requirements** - Extensions are bonus; core must work
6. **Ask Questions** - Office hours exist for a reason

---

## Example Projects from Previous Years

*(Note: These are hypothetical examples for illustration)*

- "Nowcasting U.S. GDP Using FRED-MD with Bayesian Dynamic Factors"
- "Real-Time Employment Prediction: A Sparse Factor Approach"
- "Interactive Nowcasting Dashboard for Economic Monitoring"

---

*"The goal is not just to build a model, but to build a system that could actually be used for real-time economic analysis."*

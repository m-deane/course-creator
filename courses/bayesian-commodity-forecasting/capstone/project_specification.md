# Capstone Project: End-to-End Bayesian Commodity Forecasting System

## Overview

Design, implement, and evaluate a complete Bayesian forecasting system for a commodity of your choice. This project integrates all course concepts into a production-quality analysis pipeline.

**Weight:** 30% of final grade
**Duration:** Weeks 9-12

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **Data Engineering:** Acquiring, cleaning, and preparing commodity data
2. **Bayesian Modeling:** Building appropriate probabilistic models
3. **Inference:** Running and diagnosing MCMC/VI algorithms
4. **Evaluation:** Using proper scoring rules and backtesting
5. **Communication:** Presenting technical results to stakeholders

---

## Commodity Options

Choose ONE commodity market to focus on:

### Energy
- **WTI Crude Oil** — Most liquid, extensive data, EIA fundamentals
- **Natural Gas (Henry Hub)** — Strong seasonality, storage dynamics
- **RBOB Gasoline** — Crack spread analysis, refinery dynamics

### Agriculture
- **Corn** — USDA data, strong seasonality, ethanol linkage
- **Soybeans** — Trade policy sensitivity, crush spread
- **Wheat** — Multiple varieties, global production

### Metals
- **Copper** — Industrial bellwether, China demand
- **Gold** — Safe haven, monetary policy sensitivity
- **Aluminum** — LME stocks, energy cost linkage

---

## Project Requirements

### Core Requirements (Must Complete All)

#### 1. Data Pipeline (15 points)
- [ ] Retrieve price data (minimum 5 years daily or 10 years weekly)
- [ ] Acquire fundamental data (inventories, production, consumption)
- [ ] Handle missing values appropriately
- [ ] Engineer relevant features (returns, seasonality indicators, term structure)
- [ ] Document data sources and processing steps

#### 2. Exploratory Analysis (10 points)
- [ ] Descriptive statistics and visualization
- [ ] Seasonality analysis
- [ ] Correlation with fundamentals
- [ ] Volatility dynamics (GARCH effects)
- [ ] Identify potential regime changes or structural breaks

#### 3. Bayesian Model Implementation (25 points)
Implement at least TWO of the following model types:

- [ ] **Option A:** State Space Model (local level, trend, or BSM)
- [ ] **Option B:** Hierarchical Model (if applicable, e.g., related commodities)
- [ ] **Option C:** Gaussian Process with commodity-appropriate kernel
- [ ] **Option D:** Stochastic Volatility Model
- [ ] **Option E:** Regime-Switching Model

For each model:
- Justify prior choices with domain knowledge
- Run appropriate diagnostics (R-hat, ESS, trace plots)
- Perform posterior predictive checks
- Document model specification clearly

#### 4. Forecasting & Evaluation (20 points)
- [ ] Generate rolling 1-step and multi-step forecasts
- [ ] Calculate proper scoring rules (CRPS, log score)
- [ ] Compare with naive benchmarks (random walk, seasonal naive)
- [ ] Evaluate calibration of prediction intervals
- [ ] Perform forecast combination if using multiple models

#### 5. Fundamentals Integration (15 points)
- [ ] Incorporate at least ONE fundamental variable
- [ ] Assess predictive value of fundamentals
- [ ] Discuss economic interpretation of relationships
- [ ] Address potential look-ahead bias in evaluation

#### 6. Documentation & Communication (15 points)
- [ ] Clear README with reproduction instructions
- [ ] Well-commented, organized code
- [ ] Technical report (3-5 pages, see template below)
- [ ] Final presentation (10 minutes, see rubric)

---

### Extension Options (Choose 2 for bonus points)

Each extension worth up to 5 bonus points:

1. **Real-Time Dashboard:** Create interactive visualization of forecasts
2. **Alternative Data:** Incorporate non-traditional data (satellite, shipping)
3. **Trading Strategy:** Translate forecasts into trading signals with backtested PnL
4. **Cross-Commodity:** Model relationships between 2+ related commodities
5. **Regime Analysis:** Deep dive into commodity super-cycles and regime identification
6. **Ensemble Methods:** Implement Bayesian model averaging or stacking

---

## Milestones & Checkpoints

### Milestone 1: Proposal (Week 9) — 5%
**Deliverable:** 1-page proposal

- Chosen commodity and justification
- Data sources identified
- Preliminary modeling approach
- Expected challenges

**Grading:**
- Clear, feasible scope: 2 points
- Appropriate commodity choice: 1 point
- Identified data sources: 1 point
- Realistic timeline: 1 point

### Milestone 2: Data & EDA (Week 10) — 10%
**Deliverable:** Jupyter notebook + dataset

- Complete data pipeline
- Exploratory analysis
- Feature engineering
- Preliminary findings

**Grading:**
- Data quality and completeness: 3 points
- EDA thoroughness: 3 points
- Feature engineering creativity: 2 points
- Clear documentation: 2 points

### Milestone 3: Model Implementation (Week 11) — 15%
**Deliverable:** Model code + diagnostic notebook

- At least 2 models implemented
- Convergence diagnostics passing
- Posterior predictive checks
- Preliminary forecasts

**Grading:**
- Model correctness: 5 points
- Prior justification: 3 points
- Diagnostics quality: 4 points
- Code organization: 3 points

### Milestone 4: Final Submission (Week 12) — 70%
**Deliverables:** Complete repository + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (3-5 pages, excluding figures)

1. **Executive Summary** (0.5 pages)
   - Key findings in non-technical language
   - Actionable insights

2. **Introduction & Data** (0.5-1 page)
   - Commodity market context
   - Data sources and processing
   - Key features of the data

3. **Methodology** (1-1.5 pages)
   - Model specifications with equations
   - Prior choices and justification
   - Inference approach

4. **Results** (1-1.5 pages)
   - Posterior summaries
   - Forecast performance metrics
   - Comparison across models

5. **Discussion** (0.5-1 page)
   - Economic interpretation
   - Limitations
   - Future directions

6. **References**

### Formatting
- 11pt font, 1.5 spacing
- Include figures/tables where helpful (don't count toward page limit)
- Clear section headings

---

## Presentation Rubric

### Structure (10 minutes total)
- Problem motivation: 1-2 min
- Data and EDA highlights: 2 min
- Modeling approach: 3 min
- Results and insights: 2-3 min
- Conclusion and Q&A: 1-2 min

### Evaluation Criteria

| Criterion | Excellent (4) | Good (3) | Adequate (2) | Needs Work (1) |
|-----------|--------------|----------|--------------|----------------|
| **Clarity** | Crystal clear, well-paced | Clear with minor issues | Some confusing parts | Hard to follow |
| **Technical Depth** | Demonstrates mastery | Shows solid understanding | Basic understanding | Superficial |
| **Visualization** | Outstanding, publication-quality | Good, effective | Adequate | Poor or missing |
| **Insights** | Novel, actionable insights | Useful observations | Basic conclusions | No real insights |
| **Q&A Handling** | Handles tough questions well | Answers most questions | Struggles with some | Cannot answer questions |

---

## Final Grading Rubric

### Data Pipeline (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Complete, clean data; excellent feature engineering; robust pipeline |
| 10-12 | Good data quality; solid features; minor issues |
| 7-9 | Adequate data; basic features; some gaps |
| 0-6 | Incomplete data; poor quality; major issues |

### Exploratory Analysis (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Comprehensive EDA; excellent visualizations; deep insights |
| 7-8 | Good EDA; solid plots; useful findings |
| 5-6 | Basic EDA; adequate plots; limited insights |
| 0-4 | Minimal EDA; poor visualizations |

### Model Implementation (25 points)
| Points | Criteria |
|--------|----------|
| 22-25 | Multiple well-specified models; excellent priors; passing diagnostics |
| 17-21 | Good models; reasonable priors; minor diagnostic issues |
| 12-16 | Adequate models; some prior issues; diagnostic concerns |
| 0-11 | Poor models; unjustified priors; failed diagnostics |

### Forecasting & Evaluation (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Rigorous evaluation; proper scoring rules; excellent calibration |
| 14-17 | Good evaluation; mostly proper metrics; adequate calibration |
| 10-13 | Basic evaluation; some metric issues; calibration concerns |
| 0-9 | Poor evaluation; wrong metrics; no calibration check |

### Fundamentals Integration (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Strong fundamental analysis; clear economic story; no bias |
| 10-12 | Good integration; reasonable interpretation; minor issues |
| 7-9 | Basic integration; limited interpretation; some bias concerns |
| 0-6 | Poor or missing fundamental analysis |

### Documentation & Communication (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Excellent report; outstanding presentation; reproducible code |
| 10-12 | Good report; solid presentation; mostly reproducible |
| 7-9 | Adequate report; basic presentation; some issues |
| 0-6 | Poor documentation; weak presentation; not reproducible |

---

## Academic Integrity

- This is individual work
- You may discuss concepts with peers but code must be your own
- Cite any external code or resources used
- AI assistants (ChatGPT, Copilot) may be used for debugging but not for generating solutions
- All code must be understood and explainable in Q&A

---

## Resources

### Data Sources
- **Yahoo Finance:** Futures prices (yfinance)
- **EIA.gov:** Energy data (weekly petroleum, natural gas)
- **USDA.gov:** Agricultural data (WASDE reports)
- **FRED:** Macro variables (interest rates, dollar index)
- **Quandl:** Various commodity data

### Code Templates
- See `capstone/templates/` for starter code
- Reference implementations in course notebooks

### Office Hours
- Extended office hours during capstone weeks
- Scheduled progress check-ins available

---

## Submission Instructions

1. **Create GitHub repository** with clear structure
2. **Include README.md** with setup and reproduction instructions
3. **Submit via course platform:**
   - Link to repository
   - PDF of technical report
   - PDF/link to presentation slides

---

*"The goal is not to build the best model, but to build a model you deeply understand and can defend."*

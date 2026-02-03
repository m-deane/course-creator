# Capstone Project: Market Regime Detection System

## Overview

Build a complete Hidden Markov Model system for detecting and forecasting market regimes in financial time series. Apply HMM algorithms to real market data, validate regime identification, and generate regime-aware forecasts.

**Weight:** 30% of final grade
**Duration:** Weeks 8-10

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **HMM Theory:** Understanding of Markov assumptions, state transitions, and emissions
2. **Algorithm Implementation:** Correct use of Forward-Backward, Viterbi, and Baum-Welch
3. **Model Selection:** Choosing appropriate number of states and covariance structure
4. **Financial Application:** Interpreting regimes economically and validating against market events
5. **Forecasting:** Generating regime-conditional predictions
6. **Communication:** Presenting technical results to quantitative and business audiences

---

## Project Requirements

### Core Requirements (Must Complete All)

#### 1. Data Acquisition & Preparation (15 points)

- [ ] Obtain at least 10 years of daily data for chosen market
- [ ] Compute relevant features (returns, volatility, volume, spreads, etc.)
- [ ] Handle missing data appropriately
- [ ] Create train/test split preserving temporal order
- [ ] Document data sources and processing

**Data Options:**
- **Equity Indices:** S&P 500, NASDAQ, Russell 2000
- **Commodities:** Crude oil, natural gas, gold, copper
- **Fixed Income:** Treasury yields, credit spreads
- **FX:** Major currency pairs
- **Volatility:** VIX, realized volatility

**Grading:**
- Data quality and completeness: 5 points
- Feature engineering: 5 points
- Documentation: 5 points

#### 2. Exploratory Analysis (10 points)

- [ ] Descriptive statistics and visualization
- [ ] Test for stationarity (ADF test)
- [ ] Identify potential regime changes visually
- [ ] Correlation analysis across features
- [ ] Preliminary regime hypotheses

**Grading:**
- Analysis thoroughness: 5 points
- Visualization quality: 3 points
- Economic interpretation: 2 points

#### 3. HMM Implementation (25 points)

Implement and compare at least TWO HMM specifications:

**Required:**
- [ ] Gaussian HMM with 2 states (baseline)
- [ ] Gaussian HMM with 3+ states (comparison)

**Choose ONE extension:**
- [ ] Multivariate Gaussian HMM (multiple features)
- [ ] Mixture of Gaussians emissions
- [ ] Switching ARMA emissions
- [ ] Custom emission distribution

For each model:
- [ ] Justify initialization strategy
- [ ] Run Baum-Welch until convergence
- [ ] Perform Viterbi decoding
- [ ] Compute posterior state probabilities
- [ ] Save model parameters

**Grading:**
- Model correctness: 10 points
- Implementation quality: 8 points
- Initialization strategy: 4 points
- Documentation: 3 points

#### 4. Model Selection & Diagnostics (15 points)

- [ ] Compute AIC and BIC for different K (number of states)
- [ ] Plot log-likelihood over iterations
- [ ] Validate convergence
- [ ] Check for degenerate solutions (states with very low probability)
- [ ] Justify final model selection

**Grading:**
- Selection methodology: 7 points
- Diagnostic quality: 5 points
- Justification: 3 points

#### 5. Regime Interpretation (20 points)

- [ ] Identify regimes by state parameters (means, variances)
- [ ] Provide economic labels (bull/bear, high-vol/low-vol, etc.)
- [ ] Analyze transition matrix (regime persistence, switch frequencies)
- [ ] Visualize regime classifications over time
- [ ] Validate against known market events (crashes, rallies, crises)
- [ ] Compute regime duration statistics

**Grading:**
- Economic interpretation: 8 points
- Validation against events: 7 points
- Statistical analysis: 5 points

#### 6. Out-of-Sample Forecasting (10 points)

- [ ] Forecast future states (1-step and multi-step)
- [ ] Generate regime-conditional return forecasts
- [ ] Compute forecast accuracy metrics
- [ ] Compare with regime-agnostic benchmark

**Grading:**
- Forecasting methodology: 4 points
- Evaluation rigor: 4 points
- Comparison quality: 2 points

#### 7. Documentation & Communication (5 points)

- [ ] Clear README with reproduction instructions
- [ ] Well-organized, commented code
- [ ] Technical report (see template below)
- [ ] Presentation (see rubric below)

**Grading:**
- Code quality: 2 points
- Report quality: 2 points
- Presentation: 1 point

---

### Extension Options (Choose 1-2 for Bonus)

Each extension worth up to 5 bonus points:

1. **Regime-Based Trading Strategy**
   - Develop trading rules based on regime predictions
   - Backtest with transaction costs
   - Compare Sharpe ratio to buy-and-hold

2. **Multi-Asset Regime Analysis**
   - Apply HMM to multiple related assets
   - Identify common regimes
   - Analyze regime co-movement

3. **Comparison with Alternatives**
   - Implement Markov-Switching model (statsmodels)
   - Compare with k-means clustering
   - Benchmark against structural break tests

4. **Real-Time Regime Dashboard**
   - Build interactive visualization (Streamlit/Dash)
   - Show current regime probabilities
   - Display regime-conditional statistics

5. **Advanced HMM Features**
   - Implement duration constraints
   - Add exogenous variables (fundamentals, macro)
   - Hierarchical HMM with multiple timescales

---

## Milestones & Checkpoints

### Milestone 1: Data & EDA (Week 8) — 10%
**Deliverable:** Jupyter notebook + dataset

- Complete data pipeline
- Exploratory analysis
- Preliminary regime hypotheses

**Grading:**
- Data completeness: 4 points
- EDA quality: 4 points
- Documentation: 2 points

### Milestone 2: Model Implementation (Week 9) — 15%
**Deliverable:** Model code + initial results

- At least 2 HMM specifications trained
- Convergence diagnostics
- Initial regime identification

**Grading:**
- Model correctness: 7 points
- Convergence quality: 5 points
- Code organization: 3 points

### Milestone 3: Final Submission (Week 10) — 75%
**Deliverables:** Complete analysis + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (4-6 pages, excluding figures)

1. **Executive Summary** (0.5 pages)
   - Market chosen and why
   - Key regime findings
   - Forecasting results
   - Business implications

2. **Data & Methodology** (1-1.5 pages)
   - Data sources and processing
   - Features engineered
   - HMM specifications tested
   - Model selection criteria

3. **Results** (2-2.5 pages)
   - Model selection results (AIC/BIC table)
   - Final model parameters (transition matrix, state distributions)
   - Regime interpretation and economic labeling
   - Regime timeline with key market events
   - Transition analysis

4. **Forecasting & Validation** (0.5-1 page)
   - Out-of-sample forecast methodology
   - Accuracy metrics
   - Comparison with benchmarks

5. **Discussion** (0.5-1 page)
   - Limitations
   - Practical applications
   - Future research directions

6. **Appendix**
   - Additional figures
   - Parameter tables
   - Code snippets (key algorithms)

### Key Figures to Include
- Regime classification timeline with market events
- State-conditional return distributions
- Transition probability heatmap
- Log-likelihood convergence plot
- Forecast performance comparison

---

## Presentation Rubric

### Structure (10 minutes total)
- Problem motivation: 1-2 min
- Data and methodology: 2 min
- Regime identification results: 3-4 min
- Economic interpretation: 2 min
- Conclusions and Q&A: 1-2 min

### Evaluation Criteria

| Criterion | Excellent (4) | Good (3) | Adequate (2) | Needs Work (1) |
|-----------|--------------|----------|--------------|----------------|
| **Clarity** | Crystal clear, engaging | Clear, well-paced | Understandable | Confusing |
| **Technical Rigor** | Demonstrates HMM mastery | Solid understanding | Basic grasp | Weak understanding |
| **Visualization** | Publication-quality plots | Effective figures | Adequate plots | Poor visuals |
| **Economic Insight** | Deep market understanding | Good interpretation | Basic interpretation | Superficial |
| **Q&A Handling** | Handles tough questions | Answers most questions | Struggles with some | Cannot defend work |

---

## Final Grading Rubric

### Data & Preparation (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Excellent data quality; sophisticated features; robust pipeline |
| 10-12 | Good data; solid features; minor issues |
| 7-9 | Adequate data; basic features; some gaps |
| 0-6 | Poor data quality; weak features; major issues |

### Exploratory Analysis (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Comprehensive EDA; excellent visualizations; insightful hypotheses |
| 7-8 | Good EDA; solid plots; reasonable hypotheses |
| 5-6 | Basic EDA; adequate plots; weak hypotheses |
| 0-4 | Minimal EDA; poor visualizations |

### HMM Implementation (25 points)
| Points | Criteria |
|--------|----------|
| 22-25 | Multiple correct models; excellent initialization; robust convergence |
| 17-21 | Good models; solid initialization; minor issues |
| 12-16 | Adequate models; basic initialization; convergence concerns |
| 0-11 | Incorrect models; poor initialization; failed convergence |

### Model Selection (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Rigorous selection process; comprehensive diagnostics; well-justified |
| 10-12 | Good selection; solid diagnostics; reasonable justification |
| 7-9 | Basic selection; adequate diagnostics; weak justification |
| 0-6 | Poor selection; minimal diagnostics; unjustified |

### Regime Interpretation (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Deep economic interpretation; strong validation; excellent analysis |
| 14-17 | Good interpretation; solid validation; useful analysis |
| 10-13 | Basic interpretation; some validation; limited analysis |
| 0-9 | Weak interpretation; minimal validation; poor analysis |

### Forecasting (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Rigorous forecasting; proper metrics; insightful comparison |
| 7-8 | Good forecasting; solid metrics; adequate comparison |
| 5-6 | Basic forecasting; adequate metrics; limited comparison |
| 0-4 | Poor forecasting; wrong metrics; no comparison |

### Documentation (5 points)
| Points | Criteria |
|--------|----------|
| 5 | Excellent documentation; reproducible; professional quality |
| 4 | Good documentation; mostly reproducible; solid quality |
| 3 | Adequate documentation; partially reproducible |
| 0-2 | Poor documentation; not reproducible |

---

## Technical Specifications

### Minimum Requirements
- **Python 3.8+**
- **Libraries:** hmmlearn, numpy, pandas, matplotlib, seaborn
- **Data:** Minimum 10 years daily (2,500+ observations)
- **States:** Test K = 2, 3, 4 minimum
- **Convergence:** tol=1e-2, max_iter=100 minimum

### Recommended Workflow

```python
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt

# 1. Data preparation
returns = df['close'].pct_change().dropna()
features = np.column_stack([returns, realized_vol])

# 2. Model fitting
model = hmm.GaussianHMM(n_components=2, covariance_type="full",
                         n_iter=100, tol=1e-2, random_state=42)
model.fit(features)

# 3. Regime classification
states = model.predict(features)
state_probs = model.predict_proba(features)

# 4. Analysis
print("Transition Matrix:\n", model.transmat_)
print("Means:\n", model.means_)
print("Covariances:\n", model.covars_)
```

---

## Academic Integrity

- This is individual work
- Cite any external code or resources
- You may discuss concepts but code must be your own
- Document any AI assistance used for debugging

---

## Resources

### Data Sources
- **Yahoo Finance:** Stock and index data (yfinance)
- **FRED:** Economic and financial data
- **EIA:** Energy data
- **Quandl:** Various financial data

### Software
- **hmmlearn:** [hmmlearn.readthedocs.io](https://hmmlearn.readthedocs.io)
- **pomegranate:** [pomegranate.readthedocs.io](https://pomegranate.readthedocs.io)
- **statsmodels:** For Markov-switching models

### Code Templates
- See `capstone/templates/` for starter code
- Reference implementations in course notebooks

---

## Submission Instructions

1. **Create GitHub repository** with structure:
   ```
   hmm-regime-detection/
   ├── README.md
   ├── data/
   │   ├── raw/              # Original data
   │   └── processed/        # Cleaned data
   ├── notebooks/
   │   ├── 01_data_prep.ipynb
   │   ├── 02_eda.ipynb
   │   ├── 03_hmm_models.ipynb
   │   └── 04_analysis.ipynb
   ├── src/
   │   ├── data_utils.py
   │   ├── hmm_utils.py
   │   └── visualization.py
   ├── results/
   │   ├── figures/
   │   └── models/
   ├── docs/
   │   └── report.pdf
   └── requirements.txt
   ```

2. **Submit via course platform:**
   - Link to GitHub repository
   - PDF of technical report
   - PDF of presentation slides

3. **Ensure reproducibility:**
   - requirements.txt with versions
   - Random seeds set for reproducibility
   - Clear data download/setup instructions

---

*"Markets have memory, but not infinite memory. HMMs capture this perfectly—the future depends on the present, but the present summarizes the past."*

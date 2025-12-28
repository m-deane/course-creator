# Course Syllabus: Bayesian Time Series Forecasting for Commodity Trading

## Course Information

**Course Code:** FINML-7500
**Credits:** 4
**Term:** 10-12 weeks
**Format:** Online, self-paced with weekly milestones

## Instructor

**Office Hours:** Live Q&A sessions weekly
**Response Time:** 24-48 hours on forum posts
**AI Support:** Available 24/7 for conceptual questions

---

## Course Description

This course provides a rigorous treatment of Bayesian statistical methods applied to time series forecasting in commodity markets. Students will develop both theoretical understanding and practical implementation skills, learning to build probabilistic forecasting systems that incorporate fundamental supply/demand dynamics, quantify uncertainty, and support trading decision-making.

The course emphasizes the unique characteristics of commodity markets: seasonality, storage economics, term structure dynamics, and the interplay between physical and financial markets. All concepts are grounded in real data from energy, agricultural, and metals markets.

---

## Learning Objectives

Upon successful completion of this course, students will be able to:

### Knowledge
1. Explain the Bayesian approach to statistical inference and its advantages for forecasting
2. Describe the key features of commodity markets that affect time series modeling
3. Compare different Bayesian time series models (state space, hierarchical, GPs)
4. Analyze the theoretical foundations of MCMC and variational inference

### Skills
5. Implement Bayesian models using PyMC, NumPyro, and Stan
6. Process and engineer features from commodity fundamental data
7. Diagnose model convergence and validate posterior distributions
8. Construct probabilistic forecasts with proper uncertainty quantification

### Application
9. Design forecasting systems that integrate multiple data sources
10. Evaluate forecast performance using proper scoring rules
11. Translate probabilistic forecasts into risk-adjusted trading signals
12. Communicate model results and limitations to stakeholders

---

## Prerequisites

### Required Knowledge
- **Probability & Statistics:** Random variables, distributions, expectation, variance, conditional probability, MLE
- **Linear Algebra:** Vectors, matrices, eigenvalues, matrix decomposition
- **Calculus:** Derivatives, integrals, gradients, chain rule
- **Programming:** Python fluency, NumPy/Pandas experience

### Recommended Background
- Introductory time series analysis (AR, MA, ARIMA)
- Basic understanding of financial markets
- Exposure to machine learning concepts

### Diagnostic Assessment
Complete the diagnostic quiz in Module 0 to assess readiness. Score below 70% indicates prerequisite review is needed.

---

## Course Materials

### Required Software (Free)
- Python 3.11+
- PyMC 5.x
- ArviZ
- JupyterLab

### Recommended Readings
1. **McElreath, R.** *Statistical Rethinking* (2nd ed.) - Bayesian fundamentals
2. **Durbin & Koopman** *Time Series Analysis by State Space Methods* - State space theory
3. **Gelman et al.** *Bayesian Data Analysis* (3rd ed.) - Reference text

### Data Access
- All datasets provided or accessible via free APIs
- No paid data subscriptions required

---

## Module Schedule

### Module 0: Foundations & Prerequisites (Week 0-1)
- Probability review and notation
- Python environment setup
- Introduction to commodity markets
- Diagnostic assessment

**Deliverables:** Environment setup verified, diagnostic quiz completed

### Module 1: Bayesian Fundamentals for Time Series (Week 1-2)
- Bayes' theorem and posterior inference
- Conjugate priors and analytical solutions
- Introduction to PyMC
- Simple Bayesian regression

**Deliverables:** Quiz 1, Notebook exercises

### Module 2: Commodity Market Data & Features (Week 2-3)
- Fundamental data sources (EIA, USDA, LME)
- Seasonality decomposition
- Term structure and roll dynamics
- Feature engineering for commodities

**Deliverables:** Quiz 2, Data pipeline mini-project

### Module 3: Bayesian State Space Models (Week 3-4)
- Local level and local linear trend models
- Kalman filter as Bayesian inference
- Stochastic volatility models
- Application to commodity price dynamics

**Deliverables:** Quiz 3, State space model implementation

### Module 4: Hierarchical Models for Related Commodities (Week 4-5)
- Partial pooling and shrinkage
- Cross-commodity information sharing
- Energy complex modeling (crude, products, nat gas)
- Agricultural complex (corn, soy, wheat)

**Deliverables:** Quiz 4, Hierarchical model mini-project

### Module 5: Gaussian Processes for Price Forecasting (Week 5-6)
- GP fundamentals and kernel functions
- Designing kernels for commodity seasonality
- Sparse GPs for computational efficiency
- Uncertainty quantification with GPs

**Deliverables:** Quiz 5, GP forecasting notebook

### Module 6: Inference Algorithms & Diagnostics (Week 6-7)
- MCMC foundations and Metropolis-Hastings
- Hamiltonian Monte Carlo (HMC) and NUTS
- Variational inference for scalability
- Convergence diagnostics (R-hat, ESS, trace plots)

**Deliverables:** Quiz 6, Inference comparison exercise

### Module 7: Regime Switching & Structural Breaks (Week 7-8)
- Hidden Markov models
- Bayesian change point detection
- Commodity super-cycles and regime identification
- Markov-switching stochastic volatility

**Deliverables:** Quiz 7, Regime detection mini-project

### Module 8: Fundamentals Integration & Forecast Combination (Week 8-9)
- Storage theory and convenience yield
- Supply/demand balance models
- Bayesian model averaging
- Proper scoring rules and evaluation

**Deliverables:** Quiz 8, Integrated forecasting system

### Capstone Project (Week 9-12)
- End-to-end forecasting system for chosen commodity
- Multiple checkpoints with feedback
- Final report and presentation

**Deliverables:** Proposal, checkpoint submissions, final project

---

## Assessment Details

### Weekly Quizzes (15%)
- 8 quizzes, lowest score dropped
- 15-20 questions each
- Mix of conceptual and computational
- Open book, 45-minute time limit
- Immediate feedback provided

### Coding Exercises (25%)
- Embedded in module notebooks
- Auto-graded with detailed feedback
- Unlimited attempts before deadline
- Must complete all required exercises

### Mini-Projects (30%)
- 4 bi-weekly projects
- Applied modeling tasks
- Peer review component
- Rubric-based grading

| Project | Topic | Due |
|---------|-------|-----|
| 1 | Commodity data pipeline | End of Week 3 |
| 2 | Hierarchical commodity model | End of Week 5 |
| 3 | Regime detection system | End of Week 8 |
| 4 | Forecast combination | End of Week 9 |

### Capstone Project (30%)
- Comprehensive forecasting system
- Chosen commodity market
- Multiple deliverables with checkpoints

| Milestone | Weight | Due |
|-----------|--------|-----|
| Proposal | 5% | Week 9 |
| Checkpoint 1: Data & EDA | 5% | Week 10 |
| Checkpoint 2: Model Implementation | 10% | Week 11 |
| Final Submission | 10% | Week 12 |

---

## Grading Scale

| Grade | Percentage |
|-------|------------|
| A | 93-100% |
| A- | 90-92% |
| B+ | 87-89% |
| B | 83-86% |
| B- | 80-82% |
| C+ | 77-79% |
| C | 73-76% |
| C- | 70-72% |
| F | Below 70% |

---

## Course Policies

### Late Submissions
- 10% penalty per day, up to 3 days
- No submissions accepted after 3 days without prior approval
- Extensions require documentation

### Academic Integrity
- Individual work required for quizzes and exercises
- Mini-projects allow discussion but code must be your own
- Capstone is individual work
- AI assistants (ChatGPT, Copilot) allowed for debugging, not solutions
- All code must be understood and explainable

### Collaboration Guidelines
- Encouraged: Discussing concepts, debugging help, study groups
- Prohibited: Sharing code, copying solutions, contract cheating

---

## Support Resources

### Technical Help
- Environment setup guide in `resources/`
- Common errors FAQ
- Office hours for debugging

### Content Support
- Module forums for questions
- Peer study groups
- AI chatbot for 24/7 conceptual help

### Accessibility
- All materials screen reader compatible
- Video captions provided
- Extended time available upon request
- Alternative formats available

---

## Course Updates

This syllabus may be updated to improve the learning experience. Changes will be announced in the course forum with at least one week notice for assessment-related changes.

---

*Last Updated: December 2025*

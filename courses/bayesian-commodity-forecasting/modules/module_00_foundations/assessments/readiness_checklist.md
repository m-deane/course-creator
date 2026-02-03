# Module 0: Course Readiness Checklist

## Purpose

This self-assessment helps you determine if you're ready to begin the Bayesian commodity forecasting course. Be honest with yourself - investing time in prerequisites now will save frustration later.

---

## How to Use This Checklist

For each item:
- ✅ **Confident:** You can explain this concept and apply it without reference materials
- ⚠️ **Somewhat familiar:** You've seen it before but would need to review
- ❌ **Unfamiliar:** This is new or you don't remember it

**Scoring:**
- **All ✅ or ⚠️:** Ready to proceed (review ⚠️ items as they come up)
- **1-3 ❌:** Review specific gaps using provided resources
- **4+ ❌:** Complete prerequisite coursework before starting

---

## 1. Probability Foundations

### Basic Concepts
- [ ] Understand probability as a measure of uncertainty (0 ≤ P(A) ≤ 1)
- [ ] Can calculate conditional probabilities: P(A|B)
- [ ] Know and can apply Bayes' theorem: P(A|B) = P(B|A)P(A) / P(B)
- [ ] Understand independence: P(A,B) = P(A)P(B)
- [ ] Can apply the law of total probability

### Distributions
- [ ] Know the Normal distribution properties (bell curve, defined by μ and σ)
- [ ] Understand expectation E[X] and variance Var[X]
- [ ] Can work with the Binomial distribution
- [ ] Familiar with the Exponential distribution
- [ ] Know that the Beta distribution is bounded [0,1]

### Multivariate Concepts
- [ ] Understand covariance: Cov(X,Y)
- [ ] Know correlation: ρ = Cov(X,Y) / (σ_X σ_Y)
- [ ] Can interpret a covariance matrix
- [ ] Understand the multivariate Normal distribution conceptually

---

## 2. Statistics Fundamentals

### Estimation
- [ ] Understand what a parameter is (μ, σ, β, etc.)
- [ ] Know the difference between population and sample statistics
- [ ] Can compute sample mean and variance
- [ ] Understand standard error vs standard deviation
- [ ] Familiar with confidence intervals conceptually

### Hypothesis Testing (Basic Understanding)
- [ ] Know what a p-value represents
- [ ] Understand the concept of statistical significance
- [ ] Familiar with Type I and Type II errors
- [ ] Can interpret a hypothesis test result

### Regression Basics
- [ ] Understand linear regression: y = β₀ + β₁x + ε
- [ ] Know what R² measures
- [ ] Can interpret regression coefficients
- [ ] Understand residuals and their role

---

## 3. Linear Algebra

### Vectors and Matrices
- [ ] Can perform matrix multiplication
- [ ] Understand matrix transpose
- [ ] Know what the identity matrix is
- [ ] Familiar with matrix inverse conceptually
- [ ] Can compute dot products

### Applications
- [ ] Understand vectors as representing data points
- [ ] Know that regression can be written as Y = Xβ + ε
- [ ] Familiar with covariance matrices
- [ ] Can interpret eigenvalues/eigenvectors conceptually

**Note:** You don't need to do linear algebra by hand - NumPy handles this. But you should understand what operations mean.

---

## 4. Python Programming

### Core Python
- [ ] Can write functions with arguments and return values
- [ ] Comfortable with lists, dictionaries, and loops
- [ ] Can read error messages and debug basic issues
- [ ] Familiar with virtual environments and pip

### NumPy
- [ ] Can create and manipulate NumPy arrays
- [ ] Know basic operations: sum, mean, std, reshape
- [ ] Understand broadcasting
- [ ] Can slice and index arrays

### Pandas
- [ ] Can load data into DataFrames (CSV, Excel)
- [ ] Know how to select columns and rows
- [ ] Can compute summary statistics (describe, mean, etc.)
- [ ] Familiar with time series indexing
- [ ] Can handle missing values (dropna, fillna)

### Matplotlib/Seaborn
- [ ] Can create basic line plots and histograms
- [ ] Know how to label axes and add titles
- [ ] Can customize plot appearance
- [ ] Familiar with subplots

### Jupyter Notebooks
- [ ] Can run cells and restart kernel
- [ ] Know markdown for text cells
- [ ] Can install packages from within a notebook

---

## 5. Time Series Basics (Helpful but Optional)

### Concepts
- [ ] Understand what stationarity means
- [ ] Familiar with autocorrelation
- [ ] Know what a lag is in time series context
- [ ] Can interpret a time series plot

**Note:** We'll teach time series concepts in the course, but basic familiarity helps.

---

## 6. Commodity Market Basics (Will be Covered)

You **don't** need commodity market experience to take this course. We'll teach:
- Market structure (spot vs futures)
- Fundamentals (supply, demand, inventory)
- Key data sources

But if you have domain knowledge, even better!

---

## 7. Bayesian Thinking (Will be Taught)

You **don't** need prior Bayesian experience. Module 1 starts from scratch with:
- Bayes' theorem
- Prior, likelihood, posterior
- Bayesian inference

But familiarity with the philosophy of updating beliefs with data helps.

---

## Prerequisite Resources

### If You Need Probability/Statistics Review:

**Free Resources:**
- Khan Academy: Probability & Statistics (free, excellent)
- Statistical Rethinking (Richard McElreath) - Chapters 1-3
- Think Bayes by Allen Downey (free online)

**Books:**
- "Probability and Statistics" by Morris DeGroot (comprehensive)
- "Statistical Inference" by Casella & Berger (advanced)

### If You Need Linear Algebra Review:

**Free Resources:**
- Khan Academy: Linear Algebra
- 3Blue1Brown: Essence of Linear Algebra (YouTube, visual)
- Gilbert Strang's MIT OCW Linear Algebra

**Books:**
- "Linear Algebra and Its Applications" by Gilbert Strang

### If You Need Python/Pandas Review:

**Free Resources:**
- Official pandas tutorials
- Real Python (realPython.com)
- DataCamp intro courses (first chapter free)

**Books:**
- "Python for Data Analysis" by Wes McKinney (pandas creator)
- "NumPy Cookbook" by Ivan Idris

---

## Time Investment for Prerequisite Review

| Gap Level | Estimated Time | Recommendation |
|-----------|---------------|----------------|
| **Minor (1-3 ⚠️)** | 1-2 days | Quick refresher as needed |
| **Moderate (4-6 ⚠️ or 1-2 ❌)** | 1 week | Focused review before starting |
| **Major (3+ ❌)** | 2-4 weeks | Complete foundational course first |

---

## Final Check: Diagnostic Quiz

After reviewing this checklist:

1. Complete the **Diagnostic Quiz** (diagnostic_quiz.md)
2. Work through **Probability Exercises** notebook (02_probability_exercises.ipynb)
3. Complete **Environment Setup** (01_environment_setup.ipynb)

**Pass threshold:** 70% on diagnostic quiz

---

## Ready to Proceed?

### ✅ You're ready if:
- Checklist mostly ✅ or ⚠️
- Diagnostic quiz ≥ 70%
- Environment setup runs without errors
- You're excited to learn Bayesian methods!

### ⚠️ Pause and review if:
- Multiple ❌ on checklist
- Diagnostic quiz < 70%
- Installation issues not resolved
- Feeling overwhelmed by the prerequisites

**Remember:** A strong foundation makes advanced topics easier. Don't rush.

---

## Questions?

- Post in course forum
- Attend office hours
- Email course staff

We're here to help you succeed!

---

*"Learning is not a race. Taking time to build a solid foundation is the fastest path to mastery."*

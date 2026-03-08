# Computational Complexity of Feature Selection Methods

## In Brief

Every feature selection method makes a trade-off between the quality of the solution it finds and the computational resources it consumes. Understanding the complexity of each family — in O-notation and in wall-clock terms — tells you which methods are feasible for your problem size before you write a line of code.

## Key Insight

The dominant cost in most feature selection pipelines is model training. Once you know how expensive your model is ($T_\text{model}$), you can calculate the total selection cost for each family and choose the method that fits your time budget.

---

## 1. O-Notation Primer for Feature Selection

Big-O notation describes how computational cost scales with problem size. The key variables in feature selection are:

| Symbol | Meaning | Typical range |
|---|---|---|
| $p$ | Total number of features | 10 to 100,000 |
| $n$ | Number of training samples | 100 to 10,000,000 |
| $k$ | Number of features to select | 5 to 500 |
| $T_m$ | Single model training cost | 1ms to hours |
| $G$ | Number of evolutionary generations | 20 to 500 |
| $P$ | Population size (evolutionary) | 20 to 200 |
| $p'$ | Features remaining after filter pre-screen | $k < p' \ll p$ |

**Dominant cost rule:** When analysing a pipeline, identify the step with the highest order and treat everything else as negligible.

---

## 2. Exhaustive Search: The Intractable Baseline

### 2.1 The Full Enumeration

The ideal feature selection algorithm evaluates every possible subset and returns the global optimum. The cost is:

$$T_\text{exhaustive} = O(2^p \cdot T_m)$$

There are $2^p$ binary subsets of $p$ features. For each, you train and evaluate a model.

### 2.2 Why It Is Intractable Beyond $p \approx 20$

| $p$ | Subsets | Model trains (at $T_m = 1$ ms) |
|---|---|---|
| 10 | 1,024 | 1 second |
| 15 | 32,768 | 33 seconds |
| 20 | 1,048,576 | 17 minutes |
| 25 | 33,554,432 | 9 hours |
| 30 | 1,073,741,824 | 12 days |
| 50 | $\sim 10^{15}$ | 31 million years |
| 100 | $\sim 10^{30}$ | longer than the age of the universe |

At $p = 20$ with a 1ms model, exhaustive search takes 17 minutes — feasible but barely. At $p = 25$ with even a 10ms model (a small random forest), it takes 93 days. This is why exhaustive search is a theoretical baseline, not a practical algorithm.

### 2.3 Branch-and-Bound as a Partial Fix

Branch-and-bound algorithms can prune large branches of the search tree when an upper bound on the best achievable $J(S)$ is lower than the best solution found so far. Under favourable conditions (many pruned branches), this can reduce cost significantly below $O(2^p)$, but worst-case complexity remains $O(2^p)$.

Practical limit: branch-and-bound works well for $p \leq 40$ with fast models and tight bounds.

---

## 3. Filter Methods: $O(p \cdot n)$ to $O(p^2 \cdot n)$

### 3.1 Univariate Filter: $O(p \cdot n)$

Univariate filters compute a score $s_j$ for each of the $p$ features independently. Each computation requires iterating over all $n$ samples.

**Examples and costs:**
- Pearson correlation: $O(n)$ per feature, $O(p \cdot n)$ total
- ANOVA F-statistic: $O(n)$ per feature, $O(p \cdot n)$ total
- Mutual Information (discretised): $O(n \log n)$ per feature (due to sorting), $O(p \cdot n \log n)$ total
- Chi-squared test: $O(n)$ per feature, $O(p \cdot n)$ total

**Wall-clock estimates:**
For $n = 10{,}000$ samples and an efficient NumPy implementation:

| Method | $p=100$ | $p=1{,}000$ | $p=10{,}000$ |
|---|---|---|---|
| Pearson correlation | <1 ms | <10 ms | <100 ms |
| F-statistic | <5 ms | <50 ms | <500 ms |
| Mutual Information | 50 ms | 500 ms | 5 sec |

**Univariate filters are orders of magnitude faster than any model-based method.**

### 3.2 Multivariate Filter: $O(p^2 \cdot n)$

Multivariate filters like ReliefF, mRMR, and FCBF consider feature interactions, which requires comparing feature pairs.

- **ReliefF:** $O(m \cdot k \cdot p)$ where $m$ is the number of sampled instances and $k$ is the number of neighbours. With $m = n$ and $k = 10$: $O(n \cdot 10 \cdot p) = O(10 \cdot n \cdot p)$.
- **mRMR (max-relevance min-redundancy):** Requires computing $p \times p$ pairwise MI matrix at $O(n)$ per pair: $O(p^2 \cdot n)$ total.
- **Pearson correlation matrix + VIF:** $O(p^2 \cdot n)$ for the full correlation matrix.

**Wall-clock for mRMR** ($n = 10{,}000$):

| $p=100$ | $p=1{,}000$ | $p=10{,}000$ |
|---|---|---|
| 1 sec | 100 sec | 10,000 sec (3 hours) |

For very large $p$ (> 5,000), multivariate filters become expensive. Approximate methods and random subsampling are used in practice.

---

## 4. Wrapper Methods: $O(k \cdot p \cdot T_m)$

### 4.1 Forward Selection

Forward Selection starts with $S = \emptyset$ and adds one feature at a time:

- **Step 1:** Evaluate all $p$ single-feature models. Cost: $p \cdot T_m$.
- **Step 2:** Evaluate all $p-1$ additions to the best singleton. Cost: $(p-1) \cdot T_m$.
- **Step $k$:** Evaluate all $p-k+1$ additions. Cost: $(p-k+1) \cdot T_m$.

Total cost:

$$T_\text{forward} = T_m \cdot \sum_{j=0}^{k-1} (p-j) \approx k \cdot p \cdot T_m \quad \text{for } k \ll p$$

### 4.2 Backward Elimination

Backward Elimination starts with $S = \{1,\ldots,p\}$ and removes one feature at a time:

$$T_\text{backward} = T_m \cdot \sum_{j=0}^{p-k-1} (p-j) \approx \frac{p^2}{2} \cdot T_m \quad \text{for } k \approx p/2$$

**Important:** Backward elimination is more expensive than forward selection when $p \gg k$, because it evaluates many large models before the subset shrinks.

### 4.3 RFE (Recursive Feature Elimination)

RFE trains the model on all $p$ features, removes the feature with the lowest importance, and repeats:

$$T_\text{RFE} = T_m \cdot \sum_{j=0}^{p-k-1} 1 \cdot \frac{\text{step size}}{\text{step}} \approx (p - k) \cdot T_m$$

With step size $s$ (removing $s$ features per iteration):

$$T_\text{RFE} = T_m \cdot \left\lceil \frac{p-k}{s} \right\rceil$$

RFE with cross-validation (RFECV) multiplies by the number of CV folds $f$:

$$T_\text{RFECV} = f \cdot T_m \cdot \left\lceil \frac{p-k}{s} \right\rceil$$

### 4.4 Wall-Clock Estimates for Wrapper Methods

The critical variable is $T_m$. Estimates for $k = 20$, $f = 5$ (for RFECV):

**Linear model ($T_m = 1$ ms):**

| Method | $p=100$ | $p=500$ | $p=1{,}000$ |
|---|---|---|---|
| Forward Selection | 100 ms | 500 ms | 1 sec |
| RFE (step=1) | 80 ms | 480 ms | 980 ms |
| RFECV (step=5) | 80 ms | 480 ms | 980 ms |

**Random Forest ($T_m = 1$ sec):**

| Method | $p=100$ | $p=500$ | $p=1{,}000$ |
|---|---|---|---|
| Forward Selection | 100 sec | 500 sec (8 min) | 1,000 sec (17 min) |
| RFE (step=1) | 80 sec | 480 sec (8 min) | 980 sec (16 min) |
| RFECV (step=5) | 400 sec (7 min) | 2,400 sec (40 min) | 4,900 sec (82 min) |

**LightGBM ($T_m = 5$ sec on large data):**

| Method | $p=100$ | $p=500$ | $p=1{,}000$ |
|---|---|---|---|
| Forward Selection | 8 min | 42 min | 83 min |
| RFECV (step=5) | 33 min | 3.3 hrs | 6.8 hrs |

**Rule of thumb:** If your model takes more than 1 minute to train, wrappers become painful for $p > 100$. Switch to embedded methods or hybrid (filter → wrapper with reduced $p'$).

---

## 5. Embedded Methods: $O(T_m)$ — Selection Is Free

### 5.1 The Key Property

Embedded methods integrate feature selection into model training. The selection cost is zero beyond the cost of a single model fit (with regularisation tuning).

For Lasso with coordinate descent:

$$T_\text{Lasso} = O(p \cdot n \cdot I)$$

where $I$ is the number of iterations of coordinate descent (typically 100–1,000). This is structurally the same order as the model training itself — there is no separate selection step.

**Lasso path (over a grid of $\lambda$ values):**

$$T_\text{Lasso path} = O(p \cdot n \cdot I \cdot |\Lambda|)$$

where $|\Lambda|$ is the number of regularisation values tried (e.g., 100 on a log-scale). In practice, warm-starting (using the previous $\lambda$'s solution as the initial point for the next) reduces the effective $I$ significantly.

### 5.2 Tree-Based Feature Importances

Random Forest importance computation requires no additional cost beyond training:

$$T_\text{RF importance} = O(T_\text{RF training}) = O(B \cdot n \cdot p \cdot d_\text{tree})$$

where $B$ is the number of trees and $d_\text{tree}$ is tree depth. Importance scores are a by-product of the split evaluation at no extra cost.

LightGBM and XGBoost similarly provide feature importances (gain, cover, frequency) as free by-products of gradient boosting.

### 5.3 Limitations That Affect the True Cost

While the nominal cost is $O(T_m)$, practical embedded selection requires:
- **Cross-validating the regularisation parameter $\lambda$:** adds $f \cdot |\Lambda|$ model fits for Lasso.
- **Stability selection (Meinshausen & Bühlmann, 2010):** runs Lasso on 100+ bootstrap subsamples to identify robustly selected features.

True practical cost of production-quality Lasso-based selection:

$$T_\text{Lasso, production} = f \cdot |\Lambda| \cdot T_\text{Lasso} + B_\text{bootstrap} \cdot T_\text{Lasso}$$

For $f=5$, $|\Lambda|=100$, $B=100$: 700 Lasso fits. Still much cheaper than a wrapper, since a single Lasso fit is fast.

---

## 6. Evolutionary Methods: $O(G \cdot P \cdot T_\text{fitness})$

### 6.1 The Cost Formula

Evolutionary algorithms maintain a population of $P$ candidate subsets and evolve them for $G$ generations. Each individual must be evaluated:

$$T_\text{evolutionary} = G \cdot P \cdot T_\text{fitness}$$

where $T_\text{fitness}$ is the cost of evaluating one candidate (usually a cross-validated model score).

For a GA with cross-validation ($f$ folds) as the fitness function:

$$T_\text{GA} = G \cdot P \cdot f \cdot T_m$$

### 6.2 Typical Hyperparameter Values

| Hyperparameter | Minimum | Typical | Maximum |
|---|---|---|---|
| Population $P$ | 20 | 50 | 200 |
| Generations $G$ | 20 | 100 | 500 |
| Folds $f$ | 3 | 5 | 10 |

**Total model evaluations:** $G \cdot P \cdot f = 100 \cdot 50 \cdot 5 = 25{,}000$

### 6.3 Wall-Clock Estimates

For typical GA parameters ($G=100$, $P=50$, $f=5$), total model evaluations = 25,000:

| $T_m$ | Total time | Feasibility |
|---|---|---|
| 1 ms (linear) | 25 sec | Excellent |
| 10 ms (small RF) | 4 min | Good |
| 100 ms (medium RF) | 42 min | Marginal |
| 1 sec (large RF) | 7 hours | Infeasible |
| 10 sec (LightGBM) | 70 hours | Completely infeasible |

**Practical limit for GA with 5-fold CV:** $T_m < 100$ ms, meaning linear models, small decision trees, or pre-computed fitness approximations.

### 6.4 Speedup Strategies

- **Surrogate fitness:** Train a fast proxy model (polynomial surrogate, neural network) on evaluated subsets and use it for most fitness evaluations.
- **Parallelism:** Evaluate different chromosomes in parallel across CPU/GPU cores. The $P$ evaluations per generation are embarrassingly parallel.
- **Fitness approximation:** Use 1-fold holdout instead of 5-fold CV for most generations; switch to full CV only for final evaluation.
- **Small population:** Use $P = 20$ instead of $P = 50$ for an initial exploration.
- **Transfer from filter:** Initialise the GA population using filter-ranked features rather than randomly; converges in fewer generations.

---

## 7. Comprehensive Comparison Table

### 7.1 Asymptotic Complexity

| Method | Complexity | Key variables |
|---|---|---|
| Exhaustive search | $O(2^p \cdot T_m)$ | $p$ (catastrophic growth) |
| Univariate filter | $O(p \cdot n)$ | $n, p$ |
| Multivariate filter (mRMR) | $O(p^2 \cdot n)$ | $n, p^2$ |
| Forward selection | $O(k \cdot p \cdot T_m)$ | $k, p, T_m$ |
| Backward elimination | $O(p^2 \cdot T_m)$ | $p^2, T_m$ |
| RFE | $O(p \cdot T_m)$ | $p, T_m$ |
| RFECV | $O(f \cdot p \cdot T_m)$ | $f, p, T_m$ |
| Lasso (single $\lambda$) | $O(p \cdot n \cdot I)$ | $\approx O(T_m)$ |
| Lasso (path + CV) | $O(f \cdot |\Lambda| \cdot p \cdot n \cdot I)$ | $\approx O(f \cdot |\Lambda| \cdot T_m)$ |
| Random Forest (importance) | $O(B \cdot n \cdot p \cdot d)$ | $= O(T_m)$ |
| Filter → Wrapper (hybrid) | $O(p \cdot n + k \cdot p' \cdot T_m)$ | Combines both |
| GA / PSO | $O(G \cdot P \cdot f \cdot T_m)$ | $G, P, f, T_m$ |

### 7.2 Wall-Clock Estimates: $p = 100$, $n = 10{,}000$

| Method | Linear ($T_m = 1$ ms) | RF ($T_m = 1$ s) | LightGBM ($T_m = 5$ s) |
|---|---|---|---|
| Univariate filter | <1 ms | — | — |
| Multivariate filter | 1 sec | — | — |
| Forward selection ($k=20$) | 2 sec | 33 min | 2.8 hrs |
| RFECV (step=5, $f=5$) | 2 sec | 33 min | 2.8 hrs |
| Lasso (path, 100 $\lambda$) | 100 ms | — | — |
| RF importance | — | 1 sec | — |
| GA ($G=100$, $P=50$, $f=5$) | 4 min | 35 hrs | 7.5 days |

### 7.3 Wall-Clock Estimates: $p = 1{,}000$, $n = 10{,}000$

| Method | Linear ($T_m = 1$ ms) | RF ($T_m = 1$ s) | LightGBM ($T_m = 5$ s) |
|---|---|---|---|
| Univariate filter | 10 ms | — | — |
| Multivariate filter | 1,000 sec | — | — |
| Forward selection ($k=20$) | 20 sec | 5.6 hrs | 27.8 hrs |
| RFECV (step=10, $f=5$) | 5 sec | 1.4 hrs | 6.9 hrs |
| Lasso (path, 100 $\lambda$) | 1 sec | — | — |
| RF importance | — | 1 sec | — |
| GA ($G=100$, $P=50$, $f=5$) | 4 min | 35 hrs | 7.5 days |
| Hybrid: filter($p' = 100$) + RFECV | 10 ms + 5 sec | 10 ms + 1.4 hrs | 10 ms + 6.9 hrs |

### 7.4 Wall-Clock Estimates: $p = 10{,}000$, $n = 10{,}000$

| Method | Linear ($T_m = 1$ ms) | RF ($T_m = 5$ s) | LightGBM ($T_m = 30$ s) |
|---|---|---|---|
| Univariate filter | 100 ms | — | — |
| Multivariate filter (mRMR) | ~3 hrs | — | — |
| Forward selection ($k=20$) | 3.3 min | 2.8 days | 16.7 days |
| RFECV (step=100, $f=5$) | 500 sec | 70 hrs | 17 days |
| Lasso (path, 100 $\lambda$) | 10 sec | — | — |
| RF importance | — | 5 sec | — |
| GA ($G=50$, $P=30$, $f=3$) | 1.25 min | 104 hrs | 625 hrs |
| **Hybrid: filter($p'=200$) + RFECV** | **0.1 sec + 2 sec** | **0.1 sec + 33 min** | **0.1 sec + 3.3 hrs** |

**The hybrid approach is the only practical option at $p = 10{,}000$ with slow models.**

---

## 8. Parallelism and Practical Speedups

### 8.1 Where Parallelism Helps

| Method | Parallelisable? | How |
|---|---|---|
| Univariate filter | Fully | Each feature scored independently |
| Forward Selection | Partially | Each candidate addition in a step is independent |
| RFECV | Fully (by fold) | Each CV fold is independent |
| Lasso path | Partially | Warm-starting limits benefit |
| RF importance | Fully (by tree) | Each tree in the forest is independent |
| GA/PSO | Fully (by individual) | Each chromosome evaluation is independent |

### 8.2 Practical Speedup Factors

For a machine with $C = 8$ cores:
- RFECV: up to $\min(f, C) = 5\times$ speedup (5-fold CV over 8 cores)
- GA: up to $\min(P, C) = 8\times$ speedup (population evaluated in parallel)
- Univariate filter: up to $C = 8\times$ speedup

In practice, overhead and memory contention reduce these, but 3–5× real-world speedup is achievable on an 8-core machine.

---

## 9. Choosing Based on Your Time Budget

A practical decision procedure:

1. **Estimate $T_m$:** Time a single cross-validated fit on your dataset.

2. **Calculate method costs** using the formulas above.

3. **Set your budget:** How long can the selection pipeline run? (5 min, 1 hour, overnight?)

4. **Pick the fastest method that fits within budget and the problem constraints:**

| Budget | $p$ | Recommendation |
|---|---|---|
| < 5 min | Any | Univariate or multivariate filter only |
| < 1 hr | $< 500$ | RFECV (if $T_m \cdot p \cdot f < 3600$) |
| < 1 hr | $\geq 500$ | Hybrid: filter to $p' < 200$, then RFECV |
| Overnight | $< 1000$, fast $T_m$ | GA with parallelism |
| No constraint | Small $p$, fast $T_m$ | Exhaustive or GA |

---

## Common Pitfalls

- **Forgetting cross-validation folds:** A 5-fold RFECV costs 5× more than a single RFECV pass. Always include $f$ in cost estimates.
- **Underestimating $T_m$ at scale:** A model that trains in 2 seconds on 1,000 samples may take 200 seconds on 100,000 samples. Profile $T_m$ on the actual dataset, not a subsample.
- **Ignoring mRMR's $O(p^2)$ cost:** mRMR is often described as a "filter method" (implying cheap), but its $p \times p$ pairwise MI matrix makes it as expensive as a wrapper for $p > 1{,}000$.
- **GA population too large:** Practitioners often set $P = 100$ by default. For most problems $P = 30$–$50$ with $G = 100$ is sufficient. The extra population members rarely improve the solution proportionally.

---

## Connections

- **Builds on:** Filter/wrapper/embedded/evolutionary taxonomy (Guide 01), algorithmic complexity theory
- **Leads to:** Module 01 (filter implementation), Module 03 (RFE implementation), Module 05 (GA implementation with parallel fitness evaluation)
- **Related to:** Algorithm analysis, parallel computing, Bayesian hyperparameter optimisation

---

## Further Reading

- Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *JMLR* 3, 1157–1182. — Section on wrapper complexity.
- Kohavi, R. & John, G.H. (1997). "Wrappers for Feature Subset Selection." *Artificial Intelligence* 97(1-2), 273–324. — The definitive wrapper paper.
- Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso." *JRSS-B* 58(1), 267–288. — Lasso original paper.
- Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization Paths for Generalized Linear Models via Coordinate Descent." *JSS* 33(1). — Lasso path algorithm details.

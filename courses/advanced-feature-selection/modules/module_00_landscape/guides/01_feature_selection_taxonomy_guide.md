# Feature Selection Taxonomy: A Complete Map of the Landscape

## In Brief

Feature selection identifies the most informative subset of input variables for a predictive model, discarding irrelevant and redundant features. The field divides into five families — filter, wrapper, embedded, hybrid, and evolutionary — each representing a distinct philosophy about how selection and learning interact.

## Key Insight

Every feature selection method answers the same question differently: *what makes a feature useful?* Filters use statistical independence from the target. Wrappers use model performance on held-out data. Embedded methods use the model's own learning signal. The choice of answer determines computational cost, solution quality, and practical applicability.

---

## 1. The Feature Selection Problem: Formal Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ be a dataset with $n$ samples, where $\mathbf{x}_i \in \mathbb{R}^p$ and $y_i \in \mathcal{Y}$.

The feature selection problem is a combinatorial optimisation:

$$S^* = \arg\max_{S \subseteq \{1,\ldots,p\}} J(S)$$

subject to $|S| \leq k$ for some cardinality budget $k$, where $J(S)$ is an objective function (criterion) that quantifies the utility of the subset $S$.

The search space has $2^p$ candidate subsets. For $p = 50$ that is approximately $10^{15}$ — more than the number of seconds since the Big Bang. Exhaustive search is tractable only for $p \lesssim 20$.

### What Does $J(S)$ Measure?

Different families define $J$ differently:

| Family | $J(S)$ definition |
|---|---|
| Filter | Statistical dependency $I(X_S; Y)$ or correlation |
| Wrapper | Cross-validated model accuracy $\text{CV-Acc}(\hat{f}_S)$ |
| Embedded | Regularised training loss $\mathcal{L}(\hat{f}) + \lambda \Omega(S)$ |
| Hybrid | Combination of filter pre-screening + wrapper refinement |
| Evolutionary | Any measurable fitness function (flexible) |

---

## 2. The Five Families

### 2.1 Filter Methods

Filter methods evaluate features using statistics computed directly from the data, *before* and *independently of* any learning algorithm.

**Mechanism:**
1. Compute a relevance score $s_j$ for each feature $j$ (e.g., mutual information, F-statistic, correlation coefficient).
2. Rank features by $s_j$.
3. Select the top-$k$ features or those with $s_j > \tau$ for a threshold $\tau$.

**Canonical algorithms:**
- Pearson correlation (continuous targets)
- Point-biserial correlation (binary targets)
- ANOVA F-statistic (multi-class)
- Mutual Information / Information Gain
- Chi-squared test (categorical features)
- Variance Inflation Factor (VIF) for collinearity
- ReliefF (multivariate, captures interactions)

**Strengths:**
- $O(p \cdot n)$ to $O(p^2 \cdot n)$ — scales to very large $p$
- Model-agnostic: works with any downstream learner
- No risk of overfitting the selection process
- Interpretable scores with statistical grounding

**Weaknesses:**
- Univariate filters ignore feature interactions by design
- Optimality is not guaranteed for the downstream model
- Redundancy is not handled (correlated features both pass)

**When to use:** First pass on high-dimensional data ($p > 10{,}000$), time-constrained pipelines, interpretability-critical applications.

---

### 2.2 Wrapper Methods

Wrapper methods treat the learning algorithm as a black box and use its predictive performance to evaluate subsets.

**Mechanism:**
1. Propose a feature subset $S$.
2. Train model $\hat{f}_S$ on training data restricted to $S$.
3. Evaluate $J(S) = \text{CV-Acc}(\hat{f}_S)$ on validation data.
4. Update the search strategy and repeat.

**Canonical algorithms:**
- **Forward Selection:** Start with $S = \emptyset$, greedily add the best feature at each step.
- **Backward Elimination:** Start with $S = \{1,\ldots,p\}$, greedily remove the worst feature.
- **Recursive Feature Elimination (RFE):** Train model, remove lowest-weight feature, repeat.
- **Exhaustive search:** Evaluate all $2^p$ subsets (only for small $p$).
- **Beam search:** Maintain top-$b$ partial subsets (compromise).

**Strengths:**
- Optimised directly for the target model and metric
- Naturally handles feature interactions
- Can find non-obvious subset combinations

**Weaknesses:**
- $O(k \cdot p \cdot T_\text{model})$ — expensive when $T_\text{model}$ is large
- High variance: solution depends on train/val split
- Risk of overfitting the selection process (especially forward selection on small $n$)

**When to use:** Moderate $p$ ($< 500$), fast models (linear, shallow trees), situations where you have a clear held-out validation set.

---

### 2.3 Embedded Methods

Embedded methods perform feature selection *as part of* the model training process. Selection and learning happen simultaneously.

**Mechanism:**
The model's objective function includes a regularisation term that penalises the use of features:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^n \mathcal{L}(y_i, \mathbf{x}_i^\top \boldsymbol{\beta}) + \lambda \Omega(\boldsymbol{\beta})$$

With $\Omega(\boldsymbol{\beta}) = \|\boldsymbol{\beta}\|_1$ (Lasso), zero coefficients correspond to excluded features.

**Canonical algorithms:**
- **Lasso / L1 regularisation:** Sparsity-inducing convex penalty.
- **Elastic Net:** Combines L1 (sparsity) and L2 (stability).
- **Tree-based importance:** Random Forest, Gradient Boosting feature importances.
- **Sparse Group Lasso:** Structured sparsity for grouped features.
- **Neural network pruning:** Magnitude-based or gradient-based weight zeroing.

**Strengths:**
- Computational cost is $O(T_\text{model})$ — selection comes free
- No separate validation loop needed
- Naturally handles collinear features (Lasso picks one representative)
- Captures interactions (tree-based methods)

**Weaknesses:**
- Tied to a specific model family
- Lasso can be inconsistent in the presence of correlated features
- Tree importances can be biased toward high-cardinality features
- Less interpretable selection mechanism for non-linear models

**When to use:** You have already chosen a regularisable model (Lasso, Ridge, tree ensembles), large datasets where wrapper cost is prohibitive.

---

### 2.4 Hybrid Methods

Hybrid methods chain or interleave filter and wrapper (or embedded) strategies to get the best of both worlds.

**Canonical patterns:**

**Filter-then-Wrapper (FtW):**
```
Filter phase  → Reduce p=10,000 to p'=200 (fast)
Wrapper phase → Find optimal k=20 from p'=200 (thorough)
```

**Embedded-guided Wrapper:**
Use Lasso coefficients to initialise a wrapper search (warm start).

**Multi-stage:**
1. Remove zero-variance and near-constant features (trivial filter)
2. Remove highly correlated clusters, keep one representative (correlation filter)
3. Mutual information ranking (univariate filter)
4. RFE with cross-validation (wrapper)

**Strengths:**
- Practical scalability: filter stage makes wrapper tractable
- Better solutions than pure filter, faster than pure wrapper
- Composable: stages can be independently tuned

**Weaknesses:**
- Design choices multiply: each stage has hyperparameters
- Error from early stages propagates (if filter removes a good feature, wrapper never sees it)
- Harder to reason about end-to-end behaviour

---

### 2.5 Evolutionary Methods

Evolutionary methods use population-based metaheuristics inspired by biological evolution or swarm behaviour.

**Mechanism (Genetic Algorithm example):**
1. Initialise population of $P$ binary chromosomes $\mathbf{c}^{(1)}, \ldots, \mathbf{c}^{(P)}$ where $c_j \in \{0,1\}$ indicates feature inclusion.
2. Evaluate fitness $f(\mathbf{c}) = J(S_\mathbf{c})$ for each chromosome.
3. Select parents by tournament or roulette-wheel selection.
4. Apply crossover and mutation to produce offspring.
5. Replace population. Repeat for $G$ generations.

**Canonical algorithms:**
- Genetic Algorithms (GA) — binary chromosomes
- Particle Swarm Optimisation (PSO) — continuous velocity update
- Differential Evolution (DE) — difference-vector mutation
- Ant Colony Optimisation (ACO) — pheromone-guided path
- Simulated Annealing (SA) — single-solution with temperature schedule

**Strengths:**
- Explores the full $2^p$ space without exhaustive enumeration
- Naturally escapes local optima (stochastic)
- Fitness function can be *anything* — multi-objective, non-differentiable
- Finds non-obvious, non-contiguous subsets

**Weaknesses:**
- $O(G \cdot P \cdot T_\text{fitness})$ — expensive for slow models
- No convergence guarantees; solutions vary across runs
- Many hyperparameters (population size, mutation rate, crossover type)
- Black-box: hard to explain *why* a subset was chosen

**When to use:** Complex, multi-objective problems; non-linear interactions; when no gradient is available; when global optimality matters more than speed.

---

## 3. Comparative Summary Table

| Property | Filter | Wrapper | Embedded | Hybrid | Evolutionary |
|---|---|---|---|---|---|
| **Computational cost** | $O(p \cdot n)$ | $O(k \cdot p \cdot T_m)$ | $O(T_m)$ | $O(p \cdot n + k \cdot p' \cdot T_m)$ | $O(G \cdot P \cdot T_m)$ |
| **Optimality guarantee** | None | Local (greedy) | Convex case only | None | None |
| **Scalability ($p$)** | Excellent | Poor | Good | Good | Poor |
| **Scalability ($n$)** | Good | Good | Excellent | Good | Good |
| **Handles interactions** | No (univariate) | Yes | Partial | Partial | Yes |
| **Model-agnostic** | Yes | Yes | No | Partial | Yes |
| **Overfitting risk** | Low | High | Medium | Medium | High |
| **Interpretability** | High | Medium | Medium | Medium | Low |
| **Reproducibility** | High | Medium | High | Medium | Low |

*$T_m$ = single model training cost; $k$ = number of features to select; $p'$ = features after filter; $G$ = generations; $P$ = population size.*

---

## 4. Decision Flowchart

```
START: Feature selection problem
         │
         ▼
    p > 10,000?
   /           \
 YES            NO
  │              │
  ▼              ▼
Filter      Fast model available?
(MI, F-stat)  /              \
  │         YES               NO
  │          │                 │
  │     Interpretability    Complex interactions?
  │      critical?          /              \
  │     /        \         YES              NO
  │   YES         NO        │               │
  │    │           │        ▼               ▼
  │  Filter    Embedded   Evolutionary   Wrapper
  │  only     (Lasso,     (GA, PSO)     (RFE, SFS)
  │            RF-imp)
  │
  ▼
Reduce to p' ≈ 200-500
  │
  ▼
Apply Wrapper or
Embedded on reduced set
(Hybrid approach)
```

**Quick rules:**
- Time budget < 5 min → Filter
- $p > 5{,}000$ → Filter first, then anything else
- Already using Lasso/RF → Embedded (free)
- Multi-objective or non-differentiable fitness → Evolutionary
- Everything else with moderate $p$ → Wrapper (RFE)

---

## 5. Feature Selection vs. Feature Extraction

Feature selection *retains* original features — the output is a subset of the input variables. Feature extraction *transforms* features — the output is a new representation.

| Aspect | Feature Selection | Feature Extraction |
|---|---|---|
| **Output** | Subset of original features | New transformed features |
| **Interpretability** | Preserved (original variables) | Lost (PCA components, autoencoder latents) |
| **Information retention** | May lose information | Compresses information |
| **Downstream model** | Any model | Any model |
| **Computational cost** | Varies by method | Training cost of extractor |

### When Feature Selection Wins Over PCA / Autoencoders / t-SNE

**Use feature selection when:**
1. *Interpretability is required.* Lasso coefficients and RF importances point to original business variables. PCA components are linear combinations that rarely have domain meaning.
2. *Sparsity is expected.* If only 10 of 1,000 features are truly causal, selection is more efficient than projecting all 1,000 into a dense latent space.
3. *Inference cost matters.* A deployed model using 15 features is faster and cheaper than one using all 1,000 or 50 PCA dimensions.
4. *Data is heterogeneous.* PCA assumes continuous features and linear relationships. Selection works on mixed types without transformation.
5. *Regulatory compliance.* Financial and medical models often require feature-level attribution. "This prediction uses age, income, and credit score" is auditable. "PC3 loaded on features 7, 23, 91" is not.

**Use feature extraction (PCA, autoencoders) when:**
1. All features are genuinely informative but correlated — compression beats selection.
2. You need a fixed-dimension dense representation (e.g., as input to a neural network).
3. The task is unsupervised and no target label is available to guide selection.
4. Noise reduction is the primary goal — PCA noise floor removal is well-understood.

---

## 6. The No-Free-Lunch Perspective

The No-Free-Lunch (NFL) theorems (Wolpert & Macready, 1997) state that no learning algorithm outperforms every other on all problems, averaged over all possible problem distributions. The same principle applies to feature selection.

**Practical implication:** There is no universally best feature selection method. The right choice depends on:

1. **Problem dimensionality** — $p$ and $n$ determine what is computationally feasible.
2. **Feature correlation structure** — correlated features require multivariate methods.
3. **Signal type** — linear relationships favour correlation filters; non-linear favour MI or tree-based embedded.
4. **Sample size** — wrappers require enough $n$ to estimate generalisation reliably.
5. **Downstream model** — embedded methods should match the selected model family.
6. **Computational budget** — evolutionary methods are thorough but slow.

**Evidence for NFL in practice:** Benchmarks consistently show that the best method varies by dataset. Pes et al. (2020) compared 14 filter methods across 30 datasets and found no single winner with consistent top-3 performance.

**Practical response to NFL:**
- Run multiple candidate methods in parallel.
- Use ensemble of selected subsets (stability selection).
- Cross-validate the entire selection pipeline, not just the model.
- Treat method choice as a hyperparameter to tune.

---

## Common Pitfalls

- **Selection bias:** Tuning the selection method on the test set inflates reported performance. Always nest selection inside cross-validation.
- **Univariate filter traps:** A feature with zero individual correlation to the target can still be essential in combination with other features (XOR-type interactions). Univariate filters miss this entirely.
- **Importance instability:** Random Forest importances vary across runs, especially for correlated features. Use average over multiple fits or stability selection.
- **Cardinality confusion:** Selecting exactly $k$ features is not always appropriate. Let the method determine $k$ via cross-validation.
- **Leakage through normalisation:** If you standardise features before running filter statistics, and that standardisation used the full dataset including the test fold, the selection is contaminated.

---

## Connections

- **Builds on:** Probability theory, statistical hypothesis testing, information theory, convex optimisation
- **Leads to:** Module 01 (statistical filters), Module 03 (wrappers), Module 04 (embedded methods), Module 05 (genetic algorithms)
- **Related to:** Dimensionality reduction (Module 0 extension), regularisation theory, multi-objective optimisation

---

## Further Reading

- Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." *JMLR* 3, 1157–1182. — The canonical survey; covers filter/wrapper/embedded taxonomy.
- Bolón-Canedo, V. et al. (2016). *Feature Selection for High-Dimensional Data*. Springer. — Comprehensive treatment with benchmark comparisons.
- Wolpert, D. & Macready, W. (1997). "No Free Lunch Theorems for Optimization." *IEEE TEC* 1(1), 67–82. — NFL theoretical foundation.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.), Chapter 3 (Lasso), Chapter 15 (RF importances). — Standard reference for embedded methods.

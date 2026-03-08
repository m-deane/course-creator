# Advanced Feature Selection — Glossary

All key terms used across the 12 modules. Terms are grouped by conceptual area and listed alphabetically within each group.

---

## 1. Core Feature Selection Concepts

**Backward Elimination**
A wrapper search algorithm that starts with all $p$ features and removes one feature per step — the one whose removal causes the least loss in CV performance. Cost: $O(p^2 \cdot T_m)$.

**Binary Chromosome**
In genetic algorithms, a binary string of length $p$ where each bit indicates whether the corresponding feature is included (1) or excluded (0). Represents a candidate feature subset in the $2^p$ search space.

**Boruta**
A wrapper method that uses random forest shadow features to identify "all relevant features" — not just a minimal optimal subset. Features confirmed as relevant have importances significantly above those of randomly permuted copies.

**Crossover (Genetic Algorithm)**
A genetic operator that recombines two parent chromosomes to produce offspring. Single-point crossover: choose a random cut point $c$, then child_a = parent_a[:c] + parent_b[c:].

**Curse of Dimensionality**
The phenomenon where high-dimensional spaces cause distance-based methods to degrade because pairwise distances become increasingly similar (distance concentration). The coefficient of variation of pairwise Euclidean distances approaches zero as $p \to \infty$.

**Distance Concentration**
Formally: for uniform points in $[0,1]^d$, $\lim_{d\to\infty} \text{CV}(\text{dist}) = 0$. The relative contrast $(d_\text{max} - d_\text{min})/d_\text{min}$ also collapses, making nearest-neighbour queries meaningless.

**Embedded Method**
A feature selection approach that performs selection as a side-effect of model training. Examples: Lasso (L1 regularisation drives coefficients to zero), tree-based feature importances (impurity reduction recorded at split time).

**Feature Extraction**
Transforms input features into a new representation (e.g., PCA components, autoencoder latent vectors). Unlike feature selection, the output is not a subset of original variables — interpretability is lost.

**Feature Selection**
The process of identifying a subset $S^* \subseteq \{1,\ldots,p\}$ of features that maximises a predictive utility objective $J(S)$. The search space has $2^p$ candidate subsets.

**Filter Method**
A feature selection approach that scores features using data statistics before any model is trained. Cost: $O(p \cdot n)$ for univariate filters. Examples: Pearson correlation, F-statistic, mutual information, ReliefF.

**Forward Selection**
A wrapper search algorithm that starts with no features and adds one feature per step — the one that maximises CV score. Cost: approximately $O(k \cdot p \cdot T_m)$ for selecting $k$ features from $p$.

**Hughes Phenomenon**
The empirical observation that classifier accuracy first increases and then decreases as features are added beyond the optimal subset size. Caused by the increasing proportion of noise features relative to the sample size.

**Hybrid Method**
A feature selection approach that combines two families — typically a filter pre-screen followed by a wrapper or embedded method. The filter reduces $p$ to $p' \ll p$; the wrapper works on the reduced space.

**Mutation (Genetic Algorithm)**
A genetic operator that independently flips each bit of a chromosome with probability $\mu$ (mutation rate). Prevents premature convergence by introducing genetic diversity.

**No-Free-Lunch Theorem (NFL)**
Wolpert and Macready (1997): no single optimisation algorithm outperforms random search when performance is averaged uniformly over all possible objective functions. Applied to feature selection: no method is universally best.

**p/n Ratio**
The ratio of features to samples. For $p/n > 1$ (more features than samples), the data matrix is underdetermined. Even for $p/n > 0.1$, statistical estimation becomes unreliable without regularisation.

**Parsimony**
Preference for smaller feature subsets, all else equal. Parsimony-penalised fitness: $f_p = \text{accuracy} - \alpha \cdot (k/p)$ where $\alpha$ controls the penalty strength.

**Population (Evolutionary Methods)**
The set of candidate solutions (chromosomes) maintained by an evolutionary algorithm. Population size $P$ controls the diversity of the search.

**Redundancy**
Two features are redundant if they share the same information about the target: $I(x_k; y | x_j) \approx 0$. Keeping both wastes model capacity without improving predictive performance.

**Relevance**
A feature $x_k$ is relevant to target $y$ if $I(x_k; y) > 0$ — knowing $x_k$ reduces uncertainty about $y$. Strongly relevant: $I(x_k; y | S \setminus x_k) > 0$ for all feature subsets $S$.

**RFE (Recursive Feature Elimination)**
A wrapper method that trains the model on all features, removes the feature with the lowest importance weight, and repeats. Cost: $(p-k)$ model fits for single-pass RFE; $f \cdot (p-k)/s$ for RFECV with step $s$ and $f$ folds.

**Search Space**
The set of all $2^p$ binary subsets of $p$ features. Exhaustive search is intractable for $p > 25$–$30$ with realistic model training times.

**SFFS (Sequential Floating Forward Selection)**
An extension of SFS that adds a backward phase after each forward step: if removing any currently selected feature improves the best score at the smaller subset size, that feature is removed. Escapes local optima but increases cost.

**SFS (Sequential Forward Selection)**
See Forward Selection.

**SBS (Sequential Backward Selection)**
See Backward Elimination.

**Stability Selection**
A meta-algorithm (Meinshausen and Bühlmann, 2010) that runs a selection method on bootstrap subsamples and reports selection frequencies. Features selected in more than 80–90% of runs are considered stably relevant.

**Synergy**
Two features are synergistic if they are jointly more informative than either alone: $I(x_k, x_j; y) > I(x_k; y) + I(x_j; y)$. Common example: XOR — neither $x_k$ nor $x_j$ alone predicts $y$, but their XOR does.

**T_model (Model Training Time)**
The wall-clock time for a single model fit. The dominant cost driver for wrapper and evolutionary methods. Determines feasibility: wrapper cost = $O(k \cdot p \cdot T_\text{model})$.

**Wrapper Method**
A feature selection approach that uses cross-validated model performance as the objective. Finds higher-quality feature subsets than filter methods at the cost of $O(k \cdot p \cdot T_m)$ computation.

---

## 2. Information Theory

**Conditional Mutual Information (CMI)**
$I(X; Y | Z) = H(X | Z) - H(X | Y, Z)$. The information shared by $X$ and $Y$ after accounting for $Z$. Key building block for CMIM and ICAP criteria.

**Copula**
Sklar's theorem: for any joint CDF $F(x,y)$ there exists a unique copula $C$ with $F(x,y) = C(F_X(x), F_Y(y))$. Copulas separate the marginal distributions from the dependence structure.

**Copula Mutual Information**
MI computed after transforming each variable to uniform marginals via the probability integral transform. Measures dependence structure only, removing the effect of marginal distributions.

**Distance Covariance (dCov)**
$\text{dCov}^2(X,Y) = \text{E}[A_{kl}B_{kl}]$ where $A$ and $B$ are doubly-centred pairwise distance matrices. $\text{dCov}(X,Y) = 0$ iff $X$ and $Y$ are independent (for continuous variables).

**Distance Correlation (dCor)**
$\text{dCor}(X,Y) = \sqrt{\text{dCov}^2(X,Y) / \sqrt{\text{dVar}^2(X) \cdot \text{dVar}^2(Y)}}$. Takes values in $[0,1]$; equals 0 iff independent. Detects nonlinear dependence that Pearson $r$ misses.

**Entropy (Shannon)**
$H(X) = -\mathbb{E}[\log p(X)] = -\sum_x p(x) \log p(x)$. Measures the average uncertainty (surprise) in a random variable. Units: nats (natural logarithm) or bits (log base 2).

**Histogram Estimator (MI)**
Discretise each variable into $B$ equal-width bins; estimate MI from the joint histogram. Biased downward by $O(B^2/n)$. Miller-Madow correction: add $(m-1)/(2n)$ where $m$ = non-empty joint bins.

**Interaction Information**
$\text{Int}(X;Y;Z) = I(X;Y) - I(X;Y|Z)$. Positive = redundancy (features share information). Negative = synergy (features have complementary information).

**KSG Estimator (Kraskov-Stögbauer-Grassberger)**
A k-nearest-neighbour MI estimator that avoids discretisation: $\hat{I}_\text{KSG} = \psi(k) + \psi(n) - \langle \psi(n_x+1) + \psi(n_y+1) \rangle$. Asymptotically unbiased. Implemented in sklearn's `mutual_info_classif`.

**Mutual Information (MI)**
$I(X;Y) = H(X) + H(Y) - H(X,Y) = H(X) - H(X|Y)$. Measures the reduction in uncertainty about $X$ when $Y$ is known. Always non-negative; zero iff $X$ and $Y$ are independent.

**Partial Information Decomposition (PID)**
Decomposes $I(X_1, X_2; Y)$ into four non-negative components: redundancy, unique_x1, unique_x2, and synergy. Provides a complete picture of how two features jointly inform the target.

**Rényi Entropy**
$H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_x p(x)^\alpha$. For $\alpha \to 1$, recovers Shannon entropy. $\alpha < 1$ upweights rare events (tail-sensitive); $\alpha > 1$ downweights them.

**Transfer Entropy**
$T_{X \to Y} = I(Y_{t+1}; X_{t-\ell:t} | Y_{t-k:t})$. Measures directed information flow from $X$ to $Y$ in a time series. Generalises Granger causality to nonlinear dynamics.

---

## 3. CLM Criteria Family

**Brown et al. (2012) CLM Framework**
All major ITFS criteria are special cases of:
$$J_\text{CLM}(x_k) = I(x_k; y) - \beta \sum_{x_j \in S} I(x_k; x_j) + \gamma \sum_{x_j \in S} I(x_k; x_j | y)$$
Criteria differ in $\beta$ and $\gamma$ values.

**CMIM (Conditional Mutual Information Maximisation)**
$J_\text{CMIM}(x_k) = \min_{x_j \in S} I(x_k; y | x_j)$. Greedily maximises the minimum CMI. Best criterion for datasets with feature interactions (XOR-like synergy).

**DISR (Double Input Symmetrical Relevance)**
Normalised version of JMI: $J_\text{DISR}(x_k) = \sum_{j \in S} I(x_k, x_j; y) / H(x_k, x_j, y)$. Symmetric and scale-invariant. Best for high-redundancy datasets.

**ICAP (Interaction Capping)**
$J_\text{ICAP}(x_k) = I(x_k; y) - \max(0, \sum_{x_j \in S}[I(x_k;x_j) - I(x_k;x_j|y)])$. Caps the redundancy penalty at zero to avoid over-penalising synergistic features.

**JMI (Joint Mutual Information)**
$J_\text{JMI}(x_k) = \frac{1}{|S|} \sum_{x_j \in S} I(x_k, x_j; y)$. Best average benchmark rank across datasets. Default criterion for general use.

**mRMR (Minimum Redundancy Maximum Relevance)**
$J_\text{mRMR}(x_k) = I(x_k; y) - \frac{1}{|S|} \sum_{x_j \in S} I(x_k; x_j)$. Fastest criterion ($\gamma=0$, no CMI needed). Best choice for $p > 2{,}000$.

---

## 4. Regularisation and Embedded Methods

**Coordinate Descent**
Optimisation algorithm for Lasso that updates one coefficient at a time while holding all others fixed. Complexity $O(p \cdot n \cdot I)$ where $I$ is iterations. Exploits sparsity — zero coefficients are skipped.

**ElasticNet**
Combines L1 (Lasso) and L2 (Ridge) penalties: $\hat{\beta} = \arg\min \mathcal{L} + \alpha(l_1 \|\beta\|_1 + (1-l_1)\|\beta\|_2^2)$. Handles correlated features better than Lasso alone.

**Knockoffs**
A statistical method (Candès et al., 2018) that generates fake features ("knockoffs") statistically exchangeable with originals. Features whose true importance significantly exceeds their knockoff importance are selected at a controlled FDR.

**Lasso (L1 Regularisation)**
$\hat{\beta}_\text{Lasso} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda\|\beta\|_1$. The L1 penalty drives small coefficients to exactly zero, performing selection implicitly. The regularisation path shows how the active set changes as $\lambda$ varies.

**LARS (Least Angle Regression)**
An algorithm for efficiently computing the full Lasso regularisation path. Adds features one at a time, moving in the direction equiangular between active features and the residual. Cost: $O(p^2 \cdot n)$ for the full path.

**Post-Selection Inference**
Statistical inference (confidence intervals, p-values) for coefficients after Lasso selection. Naive OLS CIs on Lasso-selected variables are anti-conservative (too narrow). Selective inference corrects for the selection event.

**RIP (Restricted Isometry Property)**
A condition on the design matrix $X$ under which Lasso recovers the true support exactly. Requires the Gram matrix of each $2k$-column submatrix to be nearly isometric. Satisfied by random Gaussian matrices with high probability.

**Stability Selection**
Run Lasso on $B = 100$ bootstrap subsamples; record the selection frequency of each feature. Features selected in $> \pi_\text{thr}$ fraction of samples are considered stable. Controls the expected number of falsely selected features.

**Structured Sparsity**
Regularisation that imposes sparsity with a known structure among features — groups, hierarchies, or graphs. Group Lasso: groups of features are either all selected or all excluded.

---

## 5. Evolutionary and Swarm Methods

**ACO (Ant Colony Optimisation)**
A population-based metaheuristic that models pheromone trails between features. Ants probabilistically select features based on pheromone levels plus heuristic relevance. Good for large, discrete search spaces.

**Chromosome**
See Binary Chromosome.

**Crossover Rate**
The probability that two selected parents undergo crossover (recombination). Typical value: 0.7–0.9. If crossover is not applied, children are copies of their parents.

**DE (Differential Evolution)**
A continuous optimisation metaheuristic adapted to feature selection via binary thresholding of continuous difference vectors. Often faster convergence than GA on continuous objective landscapes.

**Elitism**
Carrying forward the best individual(s) from one generation to the next unchanged. Prevents the best solution from being destroyed by crossover or mutation.

**Fitness Function**
The objective evaluated for each candidate solution in an evolutionary algorithm. For feature selection: typically cross-validated model accuracy, possibly penalised for subset size.

**Generation**
One iteration of an evolutionary algorithm: selection + crossover + mutation + evaluation. The algorithm runs for $G$ generations total.

**Memetic Algorithm**
Combines a global evolutionary search with a local search (hill climbing) applied to each individual. Explores globally via GA while exploiting locally via gradient descent or greedy search.

**NSGA-II**
Non-dominated Sorting Genetic Algorithm II. A multi-objective EA that maintains a Pareto front of solutions trading off accuracy vs. feature count. Uses fast non-dominated sorting and crowding distance for diversity.

**Pareto Front**
In multi-objective optimisation, the set of solutions not dominated by any other — no other solution is better on all objectives simultaneously. NSGA-II approximates the Pareto front between accuracy and parsimony.

**PSO (Particle Swarm Optimisation)**
A swarm intelligence algorithm where each particle moves through the search space guided by its personal best and the global best. Binary PSO applies a sigmoid transfer function to map continuous velocities to binary feature selections.

**Surrogate Fitness**
A cheap-to-evaluate approximation of the true fitness function. Train a surrogate (polynomial, neural network) on evaluated subsets; use it for most evaluations, reserving the true model for final candidates.

**Tournament Selection**
A selection mechanism that draws $k$ individuals at random and returns the one with the highest fitness. Tournament size $k$ controls selection pressure: $k=2$ is weak, $k=10$ is strong.

---

## 6. Time Series Feature Selection

**Granger Causality**
Tests whether lagged values of $X$ help predict $Y$ over and above $Y$'s own lags (in a linear VAR model). Feature $X$ Granger-causes $Y$ if the F-test rejects the null that lagged $X$ coefficients are zero in the $Y$ equation.

**Purged Cross-Validation**
A time-series CV scheme that removes samples from the validation set that overlap in time with the training fold (after feature computation). Prevents leakage from time-dependent feature engineering.

**Regime**
A market state or operating condition identified by a change-point or Hidden Markov Model. Regime-aware feature selection fits separate models per regime, allowing the selected feature set to vary across regimes.

**Temporal Leakage**
Information from future time points leaking into training data — e.g., standardising a time series using the global mean instead of an expanding window mean. Invalidates CV estimates and causes overoptimistic backtests.

**Transfer Entropy**
See Information Theory section.

**Walk-Forward Validation**
A time-series evaluation scheme that trains on data up to time $t$ and evaluates on data from $t$ to $t+h$. The window rolls forward in steps of $h$. Only uses past data for training — no future leakage.

---

## 7. High-Dimensional Methods

**BIC (Bayesian Information Criterion)**
$\text{BIC} = -2\log\hat{L} + k\log n$. A model selection criterion that penalises the number of parameters $k$ more heavily than AIC. Used to select the regularisation strength in Lasso.

**FDR (False Discovery Rate)**
The expected proportion of selected features that are truly null: $\text{FDR} = \mathbb{E}[\text{FP} / \max(\text{TP+FP}, 1)]$. Knockoffs and BHq control FDR at level $q$.

**FWER (Family-Wise Error Rate)**
$P(\text{at least one false positive})$. The most conservative error rate — Bonferroni correction controls FWER at level $\alpha$. More conservative than FDR control.

**Post-Selection Inference**
See Regularisation section.

**SIS (Sure Independence Screening)**
Fan and Lv (2008): rank features by marginal correlation with the target, retain the top $[n / \log n]$. Under the sure screening property, the true active set is contained in the retained set with high probability.

**Sparse PCA**
PCA variant that adds L1 regularisation to the loading vectors, producing sparse principal components. Features with non-zero loadings in at least one PC are selected.

---

## 8. Causal Feature Selection

**DAG (Directed Acyclic Graph)**
A graphical model where nodes are variables and directed edges encode direct causal relationships. $X \to Y$ means $X$ is a direct cause of $Y$.

**d-Separation**
A graphical criterion for conditional independence in a DAG. $X$ and $Y$ are d-separated given $Z$ iff all paths between them are blocked (by non-colliders in $Z$ or colliders not in $Z$). Implies $X \perp\!\!\!\perp Y | Z$.

**ICP (Invariant Causal Prediction)**
Peters, Mooij, Janzing (2016): select features whose regression coefficient is invariant across multiple experimental environments. Invariant features are causal parents of $Y$ under assumptions on the causal model.

**Intervention**
In causal analysis: setting a variable to a fixed value regardless of its natural causes, written $\text{do}(X=x)$. Feature selection based on interventional distributions finds features that are causally predictive, not just observationally correlated.

**Markov Blanket**
The minimal set of variables that renders $Y$ independent of all other variables: $Y \perp\!\!\!\perp V \setminus \text{MB}(Y) | \text{MB}(Y)$. Consists of $Y$'s parents, children, and co-parents (other parents of $Y$'s children). The optimal feature set for predicting $Y$.

**PC Algorithm**
A constraint-based causal discovery algorithm that learns a DAG skeleton from conditional independence tests, then orients edges using v-structures. Starting point for causal feature selection pipelines.

---

## 9. Production and MLOps

**Data Drift**
A change in the distribution of input features over time ($p(\mathbf{x})$ changes). Detected via statistical tests (KS test, Population Stability Index) or monitoring of feature statistics.

**Concept Drift**
A change in the relationship between features and the target ($p(y|\mathbf{x})$ changes). More dangerous than data drift — can cause silent degradation of model accuracy.

**PSI (Population Stability Index)**
$\text{PSI} = \sum_b (p_b^\text{train} - p_b^\text{new})\log(p_b^\text{train}/p_b^\text{new})$. Measures distribution shift per feature. PSI < 0.1: stable; 0.1–0.25: moderate shift; > 0.25: significant drift.

**Feature Store**
A centralised repository for computed features that ensures consistency between training and serving. Features are computed once and retrieved at training and inference time.

**MLflow**
An open-source platform for tracking ML experiments (parameters, metrics, artefacts). Used in Module 11 to log selection pipeline metadata — selected features, CV scores, and runtime.

**Pipeline (sklearn)**
A sequence of data transformation steps followed by a final estimator, implemented as `sklearn.pipeline.Pipeline`. The pipeline's `fit` method applies each step in sequence; `transform` and `predict` follow the same sequence.

**Shadow Feature**
In Boruta: a randomly shuffled copy of an original feature. Shadow features have zero true importance by construction. Boruta compares original feature importances against the maximum shadow importance.

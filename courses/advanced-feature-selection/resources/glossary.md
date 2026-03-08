# Advanced Feature Selection — Glossary

This glossary covers 150+ terms across nine categories. Each entry gives a precise definition followed by cross-references to closely related terms. Use the category headers to navigate, or search by term name.

---

## Table of Contents

1. [Statistical Methods](#1-statistical-methods)
2. [Information Theory](#2-information-theory)
3. [Wrapper Methods](#3-wrapper-methods)
4. [Embedded Methods](#4-embedded-methods)
5. [Evolutionary and Swarm Methods](#5-evolutionary-and-swarm-methods)
6. [Time Series Feature Selection](#6-time-series-feature-selection)
7. [High-Dimensional Methods](#7-high-dimensional-methods)
8. [Causal Methods](#8-causal-methods)
9. [Ensemble Selection and Production](#9-ensemble-selection-and-production)

---

## 1. Statistical Methods

**Mutual Information (MI)** — A measure of the statistical dependence between two random variables, quantifying how much knowing one variable reduces uncertainty about the other. Unlike correlation, MI captures non-linear relationships and equals zero if and only if the variables are independent. (Related: Conditional MI, Shannon Entropy, KSG Estimator)

**Conditional Mutual Information (CMI)** — The expected reduction in uncertainty about variable Y provided by variable X, given that a third variable Z is already known. CMI is the foundation of many filter methods that seek features which are informative about the target beyond what existing selected features already explain. (Related: Mutual Information, mRMR, JMI, CMIM)

**KSG Estimator** — The Kraskov–Stögbauer–Grassberger estimator, a non-parametric method for computing mutual information from continuous data using k-nearest-neighbour distances in joint and marginal spaces. It avoids binning artefacts and is consistent as sample size grows. (Related: Mutual Information, Distance Correlation)

**Distance Correlation** — A measure of dependence between two random vectors of arbitrary dimension that equals zero if and only if the variables are statistically independent. Unlike Pearson correlation, distance correlation detects non-linear and non-monotone associations. (Related: HSIC, Pearson Correlation, Mutual Information)

**Hilbert–Schmidt Independence Criterion (HSIC)** — A kernel-based measure of statistical dependence that estimates the Hilbert–Schmidt norm of the cross-covariance operator between feature and target distributions embedded in reproducing kernel Hilbert spaces. HSIC is zero if and only if the two distributions are independent. (Related: MMD, Distance Correlation, Mutual Information)

**Maximum Mean Discrepancy (MMD)** — A two-sample test statistic that measures the distance between two probability distributions as the maximum difference in expected kernel evaluations. MMD is used in feature selection to quantify distributional shift and in domain adaptation. (Related: HSIC, KL Divergence, Wasserstein Distance)

**Minimum Redundancy Maximum Relevance (mRMR)** — A filter criterion that selects features with high mutual information with the target while penalising features that are mutually redundant with already-selected features. The two scoring variants are MID (mutual information difference) and MIQ (mutual information quotient). (Related: Mutual Information, JMI, CMIM, DISR)

**MID (Mutual Information Difference)** — The mRMR variant that scores a candidate feature as its MI with the target minus the average MI between the candidate and already-selected features. MID tends to favour features with very high marginal relevance. (Related: mRMR, MIQ)

**MIQ (Mutual Information Quotient)** — The mRMR variant that scores a candidate feature as its MI with the target divided by the average MI between the candidate and already-selected features. MIQ is scale-normalised and often preferred when feature MIs span several orders of magnitude. (Related: mRMR, MID)

**Relief** — An instance-based filter algorithm that estimates feature weights by sampling instances and comparing their feature values to those of their nearest hit (same class) and nearest miss (different class). Features that distinguish hits from misses receive high weights. (Related: ReliefF, MultiSURF, FCBF)

**ReliefF** — An extension of Relief that averages over k nearest hits and k nearest misses per sampled instance and handles multi-class problems by weighting each class proportionally to its prior probability. ReliefF is robust to noise and feature interactions. (Related: Relief, MultiSURF)

**MultiSURF** — A Relief-family algorithm that adaptively selects the neighbourhood radius using the mean distance between all instance pairs, removing the need to specify k. MultiSURF identifies feature interactions more reliably than ReliefF on moderate-size datasets. (Related: ReliefF, Relief)

**Fast Correlation-Based Filter (FCBF)** — A filter method that ranks features by symmetric uncertainty with the target and then removes redundant features using a sequential pairwise comparison step. FCBF scales well to very high-dimensional data without requiring a classifier. (Related: Symmetric Uncertainty, Mutual Information, mRMR)

**Symmetric Uncertainty** — A normalised version of mutual information that scales MI by the sum of the marginal entropies of both variables, producing values in [0, 1]. Symmetric uncertainty is used in FCBF to make relevance scores comparable across features with different entropy levels. (Related: Mutual Information, FCBF, Shannon Entropy)

**Pearson Correlation** — The linear correlation coefficient between two variables, defined as their covariance divided by the product of their standard deviations. Pearson correlation only detects linear associations and assumes approximately normal distributions. (Related: Spearman Correlation, Kendall's Tau, Distance Correlation)

**Spearman Correlation** — A rank-based correlation coefficient that measures the monotone relationship between two variables by applying Pearson correlation to their rank-transformed values. Spearman is more robust to outliers than Pearson and detects any monotone association. (Related: Pearson Correlation, Kendall's Tau)

**Kendall's Tau** — A rank correlation statistic based on the proportion of concordant minus discordant pairs among all pairs of observations. Kendall's tau has a more interpretable probabilistic meaning than Spearman and is more robust to small samples. (Related: Spearman Correlation, Pearson Correlation)

**F-test** — A univariate filter that scores each feature by the ratio of between-class variance to within-class variance (ANOVA F-statistic). It tests the null hypothesis that the feature mean is identical across all classes. (Related: ANOVA F-value, Chi-squared Test, Variance Threshold)

**Chi-squared Test** — A univariate filter for categorical features that compares observed versus expected frequencies in a contingency table to test independence between the feature and the target. Higher chi-squared values indicate stronger association. (Related: F-test, Mutual Information)

**ANOVA F-value** — The test statistic from a one-way analysis of variance used in feature selection to measure how well a continuous feature discriminates between class labels. It is equivalent to the F-test statistic in the context of feature scoring. (Related: F-test, Chi-squared Test)

**Variance Threshold** — The simplest filter method, which removes any feature whose sample variance falls below a user-specified threshold. Features with near-zero variance carry little discriminative signal and can destabilise downstream models. (Related: F-test, Sure Independence Screening)

---

## 2. Information Theory

**Shannon Entropy** — The expected negative log-probability of a random variable, measuring the average amount of information (or uncertainty) in its distribution. A uniform distribution over n outcomes has maximum entropy log(n); a deterministic variable has entropy zero. (Related: Joint Entropy, Conditional Entropy, Mutual Information, Rényi Entropy)

**Joint Entropy** — The Shannon entropy of the joint distribution of two or more random variables. Joint entropy H(X, Y) equals the sum of marginal entropies minus their mutual information: H(X, Y) = H(X) + H(Y) − I(X; Y). (Related: Shannon Entropy, Conditional Entropy, Mutual Information)

**Conditional Entropy** — The average remaining uncertainty in variable Y after observing variable X, defined as H(Y | X) = H(X, Y) − H(X). Conditional entropy is always non-negative and equals zero when Y is a deterministic function of X. (Related: Shannon Entropy, Joint Entropy, Conditional MI)

**Joint Mutual Information (JMI)** — A feature-selection criterion that scores a candidate feature by its mutual information with the target conditioned on all previously selected features simultaneously. JMI provides a tighter approximation to optimal conditional selection than mRMR but is more expensive to compute. (Related: Conditional MI, CMIM, mRMR, DISR)

**Conditional Mutual Information Maximisation (CMIM)** — A greedy filter that selects the feature maximising the minimum conditional mutual information with the target given each already-selected feature. CMIM implicitly discards redundant features without estimating the joint distribution over all selected features. (Related: JMI, mRMR, Conditional MI)

**Double Input Symmetrical Relevance (DISR)** — A filter criterion based on the interaction information framework, scoring features by how much they jointly reduce uncertainty about the target beyond their individual contributions. DISR explicitly rewards synergistic feature pairs. (Related: Interaction Information, JMI, CMIM, mRMR)

**Interaction Capping (ICAP)** — A filter method that upper-bounds the redundancy penalty in the mRMR framework using pairwise conditional mutual information, producing tighter relevance estimates than mRMR at moderate computational cost. (Related: mRMR, JMI, Conditional MI)

**Rényi Entropy** — A one-parameter family of entropy measures H_α that generalises Shannon entropy, where α controls the weighting of low- versus high-probability outcomes. Shannon entropy is the limit as α → 1; Hartley entropy (α = 0) and min-entropy (α → ∞) are special cases. (Related: Shannon Entropy, KL Divergence)

**Transfer Entropy** — A directed, asymmetric measure of the information flow from one time series to another, defined as the conditional mutual information of the target's future given the source's past, conditioned on the target's own past. Transfer entropy is used in causal feature selection for time series. (Related: Granger Causality, Conditional MI, Mutual Information)

**Copula** — A multivariate distribution function that captures the dependence structure between variables independently of their marginal distributions. Copula-based MI estimators transform each variable to uniform margins, enabling distribution-free dependence estimation. (Related: Mutual Information, KSG Estimator, Distance Correlation)

**Interaction Information** — A signed measure of higher-order dependence among three or more variables that is positive when variables are redundant and negative when they are synergistic. Interaction information generalises mutual information to capture three-way feature interactions. (Related: DISR, Synergy, Redundancy, Partial Information Decomposition)

**Partial Information Decomposition (PID)** — A framework that decomposes the mutual information two sources share with a target into unique information from each source, redundant information shared by both, and synergistic information only available when both sources are observed together. (Related: Interaction Information, Synergy, Redundancy)

**Redundancy** — The component of mutual information that two features share with a target and which could be provided by either feature alone. High redundancy between selected features wastes model capacity without adding predictive signal. (Related: Synergy, Interaction Information, mRMR, Partial Information Decomposition)

**Synergy** — The component of mutual information about a target that is only accessible when two features are observed jointly, exceeding what either feature provides individually. Synergistic features may appear individually uninformative yet become powerful predictors when combined. (Related: Redundancy, Interaction Information, Partial Information Decomposition)

**Kullback–Leibler (KL) Divergence** — A measure of the directed information-theoretic distance from distribution Q to distribution P, defined as the expected log-ratio of their probability densities under P. KL divergence is asymmetric and is the basis of many MI estimation techniques. (Related: Shannon Entropy, MMD, Mutual Information)

---

## 3. Wrapper Methods

**Sequential Forward Selection (SFS)** — A greedy wrapper that starts with an empty set and iteratively adds the single feature that most improves model performance on held-out data. SFS is computationally tractable but cannot remove a feature once it has been added. (Related: Sequential Backward Selection, SFFS, Beam Search)

**Sequential Backward Selection (SBS)** — A greedy wrapper that starts with the full feature set and iteratively removes the feature whose deletion least reduces model performance. SBS is preferable when the optimal subset is large relative to the total feature count. (Related: Sequential Forward Selection, SBFS)

**Sequential Floating Forward Selection (SFFS)** — An extension of SFS that adds a backward step after each forward addition, conditionally removing features that become redundant given the newly added feature. SFFS avoids the nesting effect of pure SFS and finds better subsets at moderate extra cost. (Related: Sequential Forward Selection, SBFS)

**Sequential Floating Backward Selection (SBFS)** — An extension of SBS that adds a forward step after each backward deletion, conditionally re-adding features that become useful once a redundant feature has been removed. SBFS mirrors SFFS symmetrically. (Related: Sequential Backward Selection, SFFS)

**Boruta** — A wrapper method that tests each feature against shadow features — randomly permuted copies of all features — using a Random Forest. Features that never outperform the best shadow feature are declared unimportant; those that consistently outperform it are confirmed as relevant. (Related: Shadow Features, All-Relevant Selection, Permutation Importance)

**Shadow Features** — Randomly permuted copies of the original features used in the Boruta algorithm as a null-distribution baseline. The maximum importance score among all shadow features defines the rejection threshold for real features. (Related: Boruta, All-Relevant Selection)

**Beam Search** — A heuristic wrapper that maintains a fixed-width beam of the top-B feature subsets at each search step, expanding each subset by one feature and retaining only the best B expansions. Beam search interpolates between greedy sequential selection (B = 1) and exhaustive search. (Related: Stochastic Beam Search, Sequential Forward Selection)

**Stochastic Beam Search** — A variant of beam search that introduces randomness in the selection of which subsets to expand, improving exploration of the feature subset space at the cost of reproducibility. (Related: Beam Search, Genetic Algorithm)

**All-Relevant Selection** — The goal of identifying every feature that carries any information about the target under the full joint distribution, as opposed to finding a minimal sufficient subset. Boruta is the canonical all-relevant selector. (Related: Boruta, Minimal-Optimal Selection)

**Minimal-Optimal Selection** — The goal of finding the smallest feature subset that preserves predictive performance equal to using all features. Minimal-optimal selection is preferable when parsimony, interpretability, or inference cost matters. (Related: All-Relevant Selection, mRMR)

**Feature Pre-screening** — A fast filter step applied before a computationally expensive wrapper or embedded method to eliminate obviously uninformative features. Pre-screening reduces the search space and prevents wrappers from evaluating exponentially many useless subsets. (Related: Variance Threshold, Sure Independence Screening, Sequential Forward Selection)

---

## 4. Embedded Methods

**Lasso (L1 Regularisation)** — A linear model penalised by the L1 norm of its coefficient vector, which induces exact sparsity by shrinking many coefficients to zero. The degree of sparsity is controlled by a regularisation hyperparameter λ. (Related: Ridge, ElasticNet, LARS, Regularisation Path, Group Lasso)

**Ridge (L2 Regularisation)** — A linear model penalised by the L2 (squared) norm of its coefficient vector, which shrinks all coefficients toward zero without setting any exactly to zero. Ridge is preferred when many features have small but non-negligible effects. (Related: Lasso, ElasticNet)

**ElasticNet** — A linear model penalised by a convex combination of the L1 and L2 norms, blending Lasso's sparsity with Ridge's grouping effect for correlated features. The mixing parameter α controls the balance between the two penalties. (Related: Lasso, Ridge, Group Lasso)

**Least Angle Regression (LARS)** — An efficient algorithm for computing the entire Lasso regularisation path by moving the coefficient vector in the direction most correlated with the current residual. LARS produces the same solution path as Lasso with computational cost comparable to a single OLS fit. (Related: Lasso, Regularisation Path)

**Regularisation Path** — The sequence of model solutions obtained as the regularisation strength λ varies from zero (no penalty) to infinity (all coefficients zero). The regularisation path reveals the order in which features enter or leave the model and supports hyperparameter selection. (Related: Lasso, LARS, ElasticNet)

**Stability Selection** — A meta-algorithm that applies a sparse selection method repeatedly to bootstrap sub-samples and reports the empirical selection probability for each feature. Features selected in a large fraction of sub-samples are declared stable; selection probability thresholds can control the expected number of false discoveries. (Related: Lasso, Knockoff Filter, FDR)

**Group Lasso** — A regularisation method that applies an L2 penalty within predefined groups of features and an L1 penalty across groups, enforcing group-level sparsity while keeping all members of a selected group. It is used when features have known block structure (e.g., dummy-encoded categories). (Related: Lasso, Sparse Group Lasso, Structured Sparsity)

**Sparse Group Lasso** — A regularisation method combining Group Lasso with an additional L1 penalty on individual coefficients, achieving both group-level and within-group sparsity simultaneously. (Related: Group Lasso, Lasso, Structured Sparsity)

**Fused Lasso** — A Lasso variant that adds an L1 penalty on the differences between adjacent coefficients, encouraging piecewise-constant solutions. It is used in genomics and time series applications where features are ordered and nearby features are expected to behave similarly. (Related: Lasso, Structured Sparsity, Graph-Guided Penalty)

**Knockoff Filter** — A feature selection framework that constructs knockoff copies of features — synthetic variables that mimic the correlation structure of the originals — and uses the difference in model importance between real and knockoff features to control the false discovery rate. (Related: Model-X Knockoffs, FDR, Stability Selection)

**Model-X Knockoffs** — An extension of the knockoff filter that requires only knowledge of the joint distribution of the features (not the response distribution), making the method applicable to arbitrary non-linear models. Model-X knockoffs provide exact FDR control in finite samples. (Related: Knockoff Filter, FDR, PFER)

**False Discovery Rate (FDR)** — The expected proportion of incorrectly selected features (false positives) among all selected features. Controlling FDR at level q guarantees that on average no more than a fraction q of reported features are spurious. (Related: PFER, Knockoff Filter, Stability Selection)

**Per-Family Error Rate (PFER)** — The expected total number of false positives among all selected features, a stricter error control criterion than FDR. PFER control is appropriate when even one false positive is costly. (Related: FDR, Stability Selection, Knockoff Filter)

**Impurity Importance (Gini Importance)** — The total reduction in node impurity (typically Gini impurity or variance) attributed to a feature across all splits in a tree ensemble, averaged over trees. Impurity importance is biased toward high-cardinality features and can misrank features under class imbalance. (Related: Permutation Importance, SHAP, TreeSHAP)

**Permutation Importance** — A model-agnostic feature importance measure that quantifies the increase in prediction error when the values of a feature are randomly permuted, breaking its association with the target. Permutation importance is computed on held-out data and is less biased than impurity importance. (Related: Impurity Importance, Conditional Permutation Importance, SHAP)

**SHAP (SHapley Additive exPlanations)** — A game-theoretic framework that assigns each feature a contribution to a model's prediction equal to the Shapley value: the average marginal contribution of that feature across all possible feature orderings. SHAP values satisfy desirable axioms including efficiency, symmetry, and monotonicity. (Related: TreeSHAP, Permutation Importance, Impurity Importance)

**TreeSHAP** — A polynomial-time algorithm for computing exact SHAP values for tree-based models (decision trees, random forests, gradient boosting). TreeSHAP exploits the tree structure to reduce computation from exponential to linear in the number of features. (Related: SHAP, Impurity Importance, Permutation Importance)

**Conditional Permutation Importance** — A variant of permutation importance that permutes a feature while conditioning on its correlates, removing confounding due to feature dependencies. It provides unbiased importance estimates when features are highly correlated. (Related: Permutation Importance, SHAP)

**TabNet** — A deep learning architecture for tabular data that performs sequential attention-based feature selection at each decision step, producing instance-wise feature importance masks. TabNet trains end-to-end and selects features adaptively per prediction rather than globally. (Related: FT-Transformer, SHAP)

**FT-Transformer** — A transformer architecture adapted to tabular data by embedding each feature as a token and applying multi-head self-attention, with optional feature importance attribution via attention weights. FT-Transformer often matches or exceeds gradient boosting on tabular benchmarks with sufficient data. (Related: TabNet, SHAP)

---

## 5. Evolutionary and Swarm Methods

**Genetic Algorithm (GA)** — A population-based metaheuristic that evolves a set of candidate feature subsets (chromosomes) across generations using selection, crossover, and mutation operators. GAs explore the feature subset space broadly and are well-suited to non-linear, multi-modal fitness landscapes. (Related: Chromosome, Gene, Population, Fitness Function, NSGA-II)

**Chromosome** — In a genetic algorithm for feature selection, a binary string of length equal to the number of features, where each bit (gene) indicates whether the corresponding feature is included (1) or excluded (0). (Related: Gene, Genetic Algorithm, Mutation)

**Gene** — A single bit position in a chromosome, representing the presence or absence of one feature in the candidate subset. (Related: Chromosome, Genetic Algorithm, Bit-flip Mutation)

**Population** — The set of candidate solutions (chromosomes) maintained across generations in a genetic algorithm. A diverse population prevents premature convergence; population size controls the balance between exploration and computational cost. (Related: Genetic Algorithm, Diversity, Elitism)

**Fitness Function** — The objective function that evaluates the quality of each candidate feature subset in an evolutionary algorithm, typically a cross-validated model performance metric combined with a subset-size penalty. (Related: Genetic Algorithm, NSGA-II, Pareto Front)

**Selection Operator** — The procedure by which individuals are chosen for reproduction in a genetic algorithm, favouring higher-fitness chromosomes but retaining diversity. Common operators include tournament selection, roulette wheel selection, and rank selection. (Related: Tournament Selection, Roulette Wheel Selection, Rank Selection)

**Tournament Selection** — A selection operator that repeatedly holds a tournament among a randomly chosen subset of the population, selecting the fittest individual in each tournament. Tournament size controls selection pressure: larger tournaments increase pressure toward the fittest individuals. (Related: Roulette Wheel Selection, Rank Selection, Selection Operator)

**Roulette Wheel Selection (Fitness-Proportionate Selection)** — A selection operator that assigns each individual a probability proportional to its fitness, then samples individuals like spinning a weighted roulette wheel. It can lose diversity quickly when a few individuals dominate fitness. (Related: Tournament Selection, Rank Selection, Selection Operator)

**Rank Selection** — A selection operator that ranks all individuals by fitness and assigns selection probabilities proportional to rank rather than raw fitness, reducing selection pressure from extreme fitness outliers. (Related: Roulette Wheel Selection, Tournament Selection, Selection Operator)

**Crossover** — A reproduction operator that combines genetic material from two parent chromosomes to produce offspring, exploring new regions of the search space by recombining existing solutions. Feature selection GAs commonly use single-point and uniform crossover. (Related: Single-point Crossover, Uniform Crossover, Genetic Algorithm)

**Single-point Crossover** — A crossover operator that selects one random position in the chromosome and exchanges the tails of the two parent strings beyond that point. It preserves long contiguous feature blocks from each parent. (Related: Crossover, Uniform Crossover)

**Uniform Crossover** — A crossover operator that independently inherits each gene from one of the two parents with equal probability. Uniform crossover mixes features more thoroughly than single-point crossover and is preferred when feature order is arbitrary. (Related: Crossover, Single-point Crossover)

**Mutation** — A genetic operator that randomly flips one or more bits in a chromosome, maintaining diversity and enabling exploration of solutions not reachable by crossover alone. (Related: Bit-flip Mutation, Adaptive Mutation, Chromosome)

**Bit-flip Mutation** — The standard mutation operator for binary chromosomes: each gene is independently flipped with a small probability p_m, typically set to 1/n where n is the number of features. (Related: Mutation, Adaptive Mutation, Gene)

**Adaptive Mutation** — A mutation strategy that adjusts the mutation rate dynamically during the run, typically increasing it when population diversity falls below a threshold to escape stagnation. (Related: Mutation, Bit-flip Mutation, Premature Convergence)

**Elitism** — A strategy that copies the best-performing individual(s) from the current generation unchanged into the next, preventing the loss of the highest-quality solutions found so far. (Related: Genetic Algorithm, Convergence, Population)

**Convergence** — The state in which a population's solutions cluster near a (local) optimum, reducing the variance in fitness across individuals. Convergence is desirable when it occurs at a global optimum and undesirable when it traps the algorithm prematurely. (Related: Premature Convergence, Elitism, Diversity)

**Premature Convergence** — The collapse of population diversity to a sub-optimal solution before the global optimum has been found, typically caused by too-high selection pressure or too-low mutation rate. (Related: Convergence, Diversity, Adaptive Mutation, Fitness Sharing)

**Diversity** — A measure of the variety of solutions in the current population, quantified by metrics such as average Hamming distance between chromosomes. Maintaining diversity is critical for avoiding premature convergence. (Related: Crowding, Fitness Sharing, Niching, Population)

**Crowding** — A diversity-preservation mechanism that replaces the most similar individual in the population when a new offspring is inserted, preventing any region of the search space from being over-represented. (Related: Diversity, Fitness Sharing, Niching)

**Fitness Sharing** — A niching mechanism that reduces the fitness of individuals in densely populated regions of the search space, encouraging the population to spread across multiple optima. (Related: Niching, Crowding, Diversity)

**Niching** — A class of techniques that maintain multiple sub-populations or clusters within the main population, each converging toward a distinct local optimum. Niching supports multi-modal optimisation and feature-set ensemble construction. (Related: Fitness Sharing, Crowding, Island Model)

**Island Model** — A parallel genetic algorithm that maintains several semi-isolated sub-populations (islands), each evolving independently, with periodic migration of individuals between islands. The island model balances exploration (isolated evolution) and exploitation (migration). (Related: Niching, Genetic Algorithm, Diversity)

**NSGA-II** — Non-dominated Sorting Genetic Algorithm II, a multi-objective evolutionary algorithm that ranks solutions by non-dominated sorting and uses crowding distance as a tie-breaker to maintain a diverse Pareto front approximation. (Related: NSGA-III, MOEA/D, Pareto Front, Non-dominated Sorting, Crowding Distance)

**NSGA-III** — An extension of NSGA-II for many-objective problems (four or more objectives) that uses a set of reference points on a normalised hyperplane to guide diversity preservation, replacing the crowding distance mechanism. (Related: NSGA-II, Pareto Front, Crowding Distance)

**MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)** — A multi-objective EA that decomposes the objective space into scalar subproblems using weight vectors, solving each subproblem with assistance from neighbouring subproblems. MOEA/D is computationally efficient for large objective spaces. (Related: NSGA-II, Pareto Front)

**Pareto Front** — The set of non-dominated solutions in a multi-objective optimisation problem, where no solution can improve on one objective without worsening another. In feature selection, the Pareto front typically trades off predictive performance against subset size. (Related: Pareto Dominance, NSGA-II, Non-dominated Sorting)

**Pareto Dominance** — A partial order relation in which solution A dominates solution B if A is at least as good as B on all objectives and strictly better on at least one. Non-dominated solutions form the Pareto front. (Related: Pareto Front, NSGA-II)

**Crowding Distance** — A density estimator in NSGA-II that measures the average side length of the cuboid formed by the nearest neighbours of a solution in objective space. Higher crowding distance indicates a less crowded region, and such solutions are preferred to maintain spread along the Pareto front. (Related: NSGA-II, Pareto Front, Diversity)

**Non-dominated Sorting** — The procedure that partitions a population into successive fronts (F1, F2, ...), where F1 contains all non-dominated solutions, F2 contains solutions dominated only by F1, and so on. Used in NSGA-II and NSGA-III. (Related: Pareto Front, Pareto Dominance, NSGA-II)

**Particle Swarm Optimisation (PSO)** — A swarm intelligence algorithm in which candidate solutions (particles) move through the continuous search space guided by their own best-known position and the swarm's global best position. PSO requires a discretisation step for binary feature selection. (Related: Binary PSO, Velocity Clamping, Transfer Function, DE)

**Binary PSO** — A variant of PSO adapted for binary search spaces by interpreting each particle's velocity as a probability that the corresponding bit is set to 1, controlled via a sigmoid or V-shaped transfer function. (Related: PSO, Transfer Function, Velocity Clamping)

**Velocity Clamping** — A PSO technique that limits particle velocities to a defined range [-V_max, V_max], preventing particles from flying past the search space boundaries and ensuring stable convergence. (Related: PSO, Binary PSO)

**Transfer Function** — In Binary PSO, the function that maps a continuous velocity value to a probability in [0, 1], used to determine the probability of a binary gene being set to 1. Common choices include the S-shaped (sigmoid) and V-shaped functions. (Related: Binary PSO, PSO)

**Differential Evolution (DE)** — A population-based metaheuristic that generates trial solutions by adding the weighted difference between two randomly chosen population members to a third member, then selecting between the trial and target based on fitness. DE is effective on continuous feature weight problems. (Related: PSO, Genetic Algorithm, CMA-ES)

**Ant Colony Optimisation (ACO)** — A swarm algorithm inspired by ant foraging behaviour in which artificial ants construct solutions by traversing a construction graph, depositing pheromone on high-quality paths to guide subsequent ants. ACO is used for feature selection by encoding features as graph nodes. (Related: Pheromone, Memetic Algorithm)

**Pheromone** — In ACO, the artificial trail substance deposited on graph edges proportional to solution quality, representing the collective memory of the swarm. Higher pheromone levels attract more ants, implementing positive feedback toward good solutions. (Related: Ant Colony Optimisation)

**Memetic Algorithm** — A hybrid evolutionary algorithm that combines a population-based global search (e.g., GA) with a local search procedure applied to each individual, improving solution quality within each generation. Memetic algorithms often outperform pure EAs on feature selection. (Related: Genetic Algorithm, PSO, CMA-ES)

**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** — An adaptive evolution strategy that learns a full covariance matrix for the mutation distribution, enabling efficient search in ill-conditioned continuous spaces. CMA-ES is effective when applied to continuous feature weighting problems. (Related: Differential Evolution, Genetic Algorithm)

**Estimation of Distribution Algorithm (EDA)** — A class of evolutionary algorithms that build and sample a probabilistic model of the distribution of good solutions instead of applying crossover and mutation operators. EDAs capture variable dependencies that standard GAs cannot exploit. (Related: PBIL, Genetic Algorithm)

**Population-Based Incremental Learning (PBIL)** — An EDA that maintains a probability vector representing the marginal probability of each gene being set to 1, updating it toward high-fitness solutions and away from low-fitness solutions. PBIL is simple to implement and effective for binary feature selection. (Related: EDA, Genetic Algorithm)

**Cooperative Co-evolution** — An EA framework that decomposes the feature set into subgroups and evolves each subgroup in a separate sub-population, with fitness evaluated by combining representatives from all sub-populations. It scales to very high-dimensional feature spaces. (Related: Island Model, Genetic Algorithm)

**Surrogate-Assisted Optimisation** — An EA variant that trains a cheap surrogate model (e.g., Gaussian process) to predict fitness and uses it to pre-screen candidate solutions, reserving expensive true fitness evaluations for promising individuals. Critical when each fitness evaluation requires full model training. (Related: Genetic Algorithm, Memetic Algorithm, AutoML)

---

## 6. Time Series Feature Selection

**Granger Causality** — A statistical test asserting that time series X Granger-causes Y if past values of X improve prediction of Y beyond what Y's own past provides. Granger causality is a predictive notion, not a structural causal claim, and is confounded by omitted common drivers. (Related: Conditional Granger Causality, Transfer Entropy)

**Conditional Granger Causality** — An extension of Granger causality that tests whether X improves prediction of Y after conditioning on a third set of variables Z, reducing spurious causality attributions from common drivers. (Related: Granger Causality, Transfer Entropy)

**Spectral Granger Causality** — A frequency-domain decomposition of Granger causality that attributes causal influence to specific frequency bands, useful for identifying features that drive the target at particular periodicities. (Related: Granger Causality, Autocorrelation)

**Walk-Forward Validation** — A time-series cross-validation protocol that trains on all data up to time t and evaluates on a fixed window starting at t + gap, then advances t by one step. It respects temporal ordering and prevents data leakage from future to past. (Related: Purged Cross-Validation, Embargo, Combinatorial Purged CV)

**Purged Cross-Validation** — A cross-validation scheme for financial or sequential data that removes (purges) training observations whose labels overlap in time with any observation in the validation fold, eliminating label leakage across the train-test boundary. (Related: Walk-Forward Validation, Embargo)

**Embargo** — A gap period imposed between the training and validation sets in purged cross-validation to prevent information leakage via serial correlation in the features, even after purging overlapping labels. (Related: Purged Cross-Validation, Walk-Forward Validation)

**Combinatorial Purged Cross-Validation (CPCV)** — A variant of purged cross-validation that uses combinatorial splits to generate a larger number of test paths, providing lower-variance estimates of out-of-sample performance for financial time series. (Related: Purged Cross-Validation, Walk-Forward Validation)

**Feature Drift** — The change over time in the statistical distribution of a feature in production, which can degrade model performance even when the feature–target relationship is stable. Feature drift is monitored via PSI, KS tests, and Wasserstein distance. (Related: Concept Drift, PSI, KS Test, Wasserstein Distance)

**Regime** — A persistent state of a time series characterised by a distinct statistical pattern, such as a bull or bear market, high or low volatility period, or expansion/recession cycle. Feature relevance often varies across regimes. (Related: HMM, Markov Switching, Change Point Detection)

**Hidden Markov Model (HMM)** — A probabilistic model of a time series that assumes observations are generated from a Markov chain of latent states (regimes), with each state having its own emission distribution. HMMs are used both to identify regimes and as a feature engineering tool. (Related: Regime, Markov Switching, Change Point Detection)

**Markov Switching Model** — An econometric model in which parameters (e.g., mean, variance, regression coefficients) switch between a finite number of states according to a Markov chain. It generalises standard time series models to accommodate structural breaks and regime changes. (Related: HMM, Regime, Change Point Detection)

**Change Point Detection** — Methods for identifying time points at which the statistical properties of a time series change abruptly, including shifts in mean, variance, or autocovariance structure. Detected change points can be used to segment features or define regime indicators. (Related: Regime, HMM, Stationarity)

**Stationarity** — The property of a time series whose statistical properties (mean, variance, autocovariance) do not change over time. Feature selection for time series typically requires stationarity; non-stationary features are differenced or detrended before use. (Related: ADF Test, KPSS Test, Change Point Detection)

**Augmented Dickey–Fuller (ADF) Test** — A unit-root test that examines the null hypothesis that a time series has a unit root (is integrated of order one), against the alternative of stationarity. Failing to reject the null indicates the feature may need differencing. (Related: KPSS Test, Stationarity)

**KPSS Test** — The Kwiatkowski–Phillips–Schmidt–Shin test, which reverses the ADF null hypothesis by testing stationarity against the alternative of a unit root. Using ADF and KPSS together provides a more reliable assessment of integration order. (Related: ADF Test, Stationarity)

**Lag Selection** — The problem of determining which past values (lags) of a time series carry predictive information about the current value. Lag selection affects autocorrelation structure and can dramatically change the number of derived features. (Related: Autocorrelation, Granger Causality)

**Autocorrelation** — The correlation of a time series with its own past values at various lags. The autocorrelation function (ACF) is used to diagnose temporal dependence in residuals and to motivate lag feature engineering. (Related: Lag Selection, Cross-correlation, Stationarity)

**Cross-correlation** — The correlation between two time series at various lags, used to detect lead-lag relationships between a feature and the target. Significant cross-correlation at positive lags suggests a feature may Granger-cause the target. (Related: Autocorrelation, Granger Causality, Lag Selection)

**Rolling Window** — A sliding computation window of fixed length moved across a time series to compute local statistics (mean, variance, correlation) as features. Rolling window features capture recent regime characteristics without relying on full-history stationarity. (Related: Feature Drift, Lag Selection, Regime)

---

## 7. High-Dimensional Methods

**Sure Independence Screening (SIS)** — A marginal correlation screening procedure that ranks features by their marginal correlation with the target and retains the top n features, where n is of the same order as the sample size. Under regularity conditions SIS is guaranteed to retain all truly relevant features — the sure screening property. (Related: ISIS, DC-SIS, Sure Screening Property, Variance Threshold)

**Iterative Sure Independence Screening (ISIS)** — An extension of SIS that performs multiple rounds of screening, fitting a sparse model on the screened features and iteratively adding features that explain residual variation, overcoming SIS's failure when relevant features are marginally uncorrelated with the target. (Related: SIS, Sure Screening Property)

**Distance Correlation SIS (DC-SIS)** — A variant of SIS that replaces Pearson marginal correlation with distance correlation, enabling screening of non-linear feature–target associations in ultra-high-dimensional settings. (Related: SIS, Distance Correlation, Sure Screening Property)

**Sure Screening Property** — The theoretical guarantee that a screening procedure retains all truly relevant features in the reduced set with probability approaching one as sample size grows, provided certain regularity conditions hold. (Related: SIS, ISIS, DC-SIS)

**Sparse PCA** — A variant of principal component analysis that constrains the loading vectors to be sparse, producing interpretable components that are linear combinations of only a small subset of features. (Related: Random Projection, Structured Sparsity, Lasso)

**Random Projection** — A dimensionality reduction technique that multiplies the feature matrix by a random matrix with orthonormal or sub-Gaussian columns, approximately preserving pairwise distances per the Johnson–Lindenstrauss Lemma. Used as a pre-processing step before feature selection in very high dimensions. (Related: Johnson–Lindenstrauss Lemma, Sparse PCA)

**Johnson–Lindenstrauss Lemma** — A mathematical result stating that any set of n points in high-dimensional space can be embedded into O(log n / ε²) dimensions with pairwise distances distorted by at most a factor of (1 ± ε), using a random linear projection. (Related: Random Projection)

**Debiased Lasso** — A post-selection inference technique that corrects the Lasso coefficient estimates for the regularisation bias, enabling construction of valid confidence intervals and hypothesis tests after Lasso model selection. (Related: Lasso, Post-Selection Inference, Selective Inference)

**Selective Inference** — Statistical inference that accounts for the fact that the hypothesis being tested was selected by examining the data, providing valid p-values and confidence intervals conditional on the selection event. (Related: Post-Selection Inference, Debiased Lasso)

**Post-Selection Inference** — The broad problem of drawing valid statistical conclusions about model parameters or feature relevance after a data-driven model selection step. Naive inference ignores selection bias and produces anti-conservative p-values. (Related: Selective Inference, Debiased Lasso, FDR)

**Compact Genetic Algorithm (cGA)** — An EDA that represents the population implicitly as a probability vector updated by comparing two competing solutions, enabling efficient large-scale binary optimisation with low memory requirements. (Related: PBIL, EDA, Genetic Algorithm)

**Structured Sparsity** — Sparsity patterns constrained to conform to prior knowledge about feature organisation, such as group sparsity, tree sparsity, or graph sparsity. Structured sparsity penalties encode domain knowledge about which features should be selected together. (Related: Group Lasso, Fused Lasso, Graph-Guided Penalty, Sparse PCA)

**Graph-Guided Penalty** — A regularisation method that encodes a feature dependency graph into the penalty term, encouraging co-selection of features connected by edges (e.g., via a Laplacian penalty). Used in genomics to leverage biological pathway information. (Related: Structured Sparsity, Fused Lasso, Group Lasso)

---

## 8. Causal Methods

**Causal Graph** — A directed graph in which nodes represent variables and edges represent direct causal influences. Causal graphs make explicit the structural assumptions underlying a causal model and support identification of causal effects via do-calculus. (Related: DAG, Structural Causal Model, PC Algorithm)

**Directed Acyclic Graph (DAG)** — A directed graph with no directed cycles, used to represent causal structure among variables. In a causal DAG, the absence of an edge between two nodes encodes conditional independence. (Related: Causal Graph, Structural Causal Model, Markov Blanket)

**Structural Causal Model (SCM)** — A formal representation of a data-generating process using a set of structural equations, one per variable, that specify each variable as a deterministic function of its parents in the causal DAG plus an independent noise term. SCMs support interventional and counterfactual reasoning. (Related: DAG, do-Calculus, Causal Graph)

**do-Calculus** — Pearl's set of three graphical inference rules (rules of deletion, action, and observation) that allow identification and estimation of interventional distributions P(Y | do(X)) from observational data and a causal DAG. (Related: Structural Causal Model, Causal Graph, Invariant Causal Prediction)

**Markov Blanket** — The minimal set of variables that renders a target variable conditionally independent of all other variables in a Bayesian network: the target's parents, children, and co-parents. The Markov blanket is the theoretically optimal feature subset for predicting the target. (Related: DAG, PC Algorithm, FCI)

**PC Algorithm** — A constraint-based causal discovery algorithm that recovers the Markov equivalence class of the true DAG by testing conditional independences and orienting edges using Meek's orientation rules. The PC algorithm is sound and complete under faithfulness and the Markov condition. (Related: FCI, GES, DAG, Markov Blanket)

**Fast Causal Inference (FCI)** — An extension of the PC algorithm that handles latent confounders and selection bias, producing a partial ancestral graph (PAG) rather than a DAG, with additional edge types representing the presence of latent common causes. (Related: PC Algorithm, GES, Confounder)

**Greedy Equivalence Search (GES)** — A score-based causal discovery algorithm that searches the space of Markov equivalence classes using a greedy forward-backward strategy, adding and then removing edges to maximise a decomposable score such as BIC. (Related: PC Algorithm, FCI, DAG)

**Invariant Causal Prediction (ICP)** — A causal feature selection method that identifies the subset of features whose linear relationship with the target is invariant across different experimental environments. Features in the invariant set are interpreted as direct causes of the target. (Related: Environment, Causal Graph, Double/Debiased ML)

**Environment** — In causal and distributional robustness methods, a distinct data-generating context (e.g., different time period, country, intervention level) under which the joint feature–target distribution may differ. ICP and domain generalisation methods exploit multiple environments to identify causal features. (Related: Invariant Causal Prediction, Concept Drift, Regime)

**Double/Debiased Machine Learning (DML)** — A framework for semi-parametric causal effect estimation that uses cross-fitted ML models to partial out the effects of high-dimensional nuisance variables, achieving root-n consistent and asymptotically normal estimates of target parameters. (Related: Causal Forest, Instrumental Variable, Confounder)

**Causal Forest** — A non-parametric method for estimating heterogeneous treatment effects by adapting random forests to target local causal effect estimates, using an honest splitting criterion to separate estimation from model selection. (Related: Double/Debiased ML, Instrumental Variable)

**Instrumental Variable (IV)** — A variable that is correlated with the treatment variable but affects the outcome only through the treatment, allowing identification of the causal effect of the treatment in the presence of unobserved confounders. (Related: Confounder, Double/Debiased ML, Causal Forest)

**Confounder** — A variable that causally influences both a feature of interest and the target outcome, creating a spurious association between feature and target that is not mediated by a direct causal path. Failing to adjust for confounders leads to biased importance estimates. (Related: Collider, Instrumental Variable, Causal Graph)

**Collider** — A variable that is caused by two or more other variables in a causal graph. Conditioning on a collider opens a spurious association between its parents (Berkson's paradox), a common source of bias when filtering features on post-treatment variables. (Related: Confounder, DAG, Causal Graph)

**Faithfulness** — The assumption that all conditional independences in the observed data are entailed by the causal graph via d-separation, and that no conditional independence arises from exact cancellations of path coefficients. Faithfulness is required for identifiability in constraint-based causal discovery. (Related: PC Algorithm, DAG, Markov Blanket)

---

## 9. Ensemble Selection and Production

**Ensemble Feature Selection** — The practice of aggregating feature importance rankings or selection indicators from multiple base selectors (different algorithms, datasets, or parameters) to produce a more stable and accurate final feature set. (Related: Kuncheva's Index, Borda Count, Rank Aggregation)

**Kuncheva's Index** — A stability measure for feature selection that quantifies the similarity between two selected subsets of size k from n features, corrected for chance agreement. A value near 1 indicates perfect stability; near 0 indicates random selection. (Related: Jaccard Similarity, Ensemble Feature Selection)

**Jaccard Similarity** — The ratio of the intersection to the union of two selected feature sets, used as a simple pairwise stability measure. Jaccard similarity is intuitive but not corrected for the effect of subset size. (Related: Kuncheva's Index, Ensemble Feature Selection)

**Borda Count** — A rank aggregation method that assigns each feature a score equal to the number of other features ranked below it in each ranker's ordering, then sums scores across rankers. The Borda count provides a consensus ranking from multiple feature rankers. (Related: Rank Aggregation, Kemeny Optimal, Ensemble Feature Selection)

**Rank Aggregation** — The problem of combining multiple ranked lists of features into a single consensus ranking. Methods include Borda count, Kemeny optimal aggregation, and Spearman footrule minimisation. (Related: Borda Count, Kemeny Optimal)

**Kemeny Optimal Aggregation** — The rank aggregation method that minimises the total pairwise disagreement (Kemeny distance) between the consensus ranking and all input rankings. It is NP-hard in general but solvable exactly for moderate feature counts. (Related: Rank Aggregation, Borda Count)

**Meta-Learning** — The use of dataset-level features (meta-features) to predict which feature selection algorithm or configuration will perform best on a new dataset, without running all algorithms explicitly. Meta-learning underpins AutoML recommender systems. (Related: Dataset Meta-Features, AutoML, CASH Problem)

**Dataset Meta-Features** — Descriptive statistics of a dataset used as inputs to meta-learning models, including number of features, class imbalance ratio, feature entropy distributions, kurtosis, and estimated intrinsic dimensionality. (Related: Meta-Learning, AutoML)

**AutoML** — Automated machine learning systems that search the combined algorithm selection and hyperparameter optimisation (CASH) problem, including automated feature selection, pipeline construction, and model evaluation. (Related: CASH Problem, Meta-Learning, Surrogate-Assisted Optimisation)

**CASH Problem (Combined Algorithm Selection and Hyperparameter Optimisation)** — The joint optimisation problem of selecting the best learning algorithm and its hyperparameters from a large combined configuration space. Feature selection is typically embedded as a pipeline stage within the CASH search. (Related: AutoML, Meta-Learning)

**Bayesian Model Averaging (BMA)** — A technique that averages predictions across multiple feature subsets or models weighted by their posterior probability, reducing model selection uncertainty. BMA produces well-calibrated predictions when the true model is in the candidate set. (Related: Ensemble Feature Selection, Stability Selection)

**Feature Store** — A centralised data infrastructure component that manages the computation, storage, versioning, and serving of features for machine learning models, ensuring consistency between training and serving pipelines. (Related: MLflow, Reproducibility, Audit Trail)

**Population Stability Index (PSI)** — A monitoring statistic that measures the distributional shift of a feature (or model score) between a reference period and a current production period by summing the product of bin proportion differences and log-ratios. PSI > 0.25 conventionally triggers a feature review. (Related: Feature Drift, KS Test, Wasserstein Distance)

**Kolmogorov–Smirnov (KS) Test** — A non-parametric test that quantifies the maximum absolute difference between two empirical cumulative distribution functions, used to detect distributional shift in features between training and production data. (Related: PSI, Wasserstein Distance, Feature Drift)

**Wasserstein Distance** — The optimal transport distance between two probability distributions, interpretable as the minimum expected cost of transforming one distribution into the other. Wasserstein distance is more sensitive to tail differences than KS and PSI. (Related: KS Test, PSI, MMD, Feature Drift)

**Concept Drift** — The change over time in the conditional distribution of the target given the features, P(Y | X), meaning the underlying predictive relationship has shifted even if feature distributions remain stable. Concept drift requires model retraining rather than feature re-selection alone. (Related: Feature Drift, Regime, Change Point Detection)

**A/B Testing** — A controlled experiment design that randomly assigns observations to control and treatment groups to estimate the causal effect of an intervention (e.g., a new feature set) on a target metric. A/B testing provides unbiased comparison of feature pipelines in production. (Related: Instrumental Variable, Double/Debiased ML)

**MLflow** — An open-source platform for managing the machine learning lifecycle, including experiment tracking, model versioning, and deployment. MLflow enables reproducibility and audit trails for feature selection experiments. (Related: Feature Store, Reproducibility, Audit Trail)

**Reproducibility** — The ability to re-run a feature selection experiment and obtain the same results, requiring fixed random seeds, versioned data, pinned library dependencies, and logged hyperparameters. Reproducibility is a prerequisite for scientific validity and production reliability. (Related: MLflow, Audit Trail, Feature Store)

**Audit Trail** — A complete, tamper-evident log of all decisions, data versions, code versions, and parameter choices used in a feature selection pipeline, enabling post-hoc inspection and regulatory compliance. Audit trails are required in regulated industries such as finance and healthcare. (Related: MLflow, Reproducibility, Feature Store)

---

*Last updated: 2026-03-08. This glossary covers the core vocabulary of the Advanced Feature Selection course. For notation conventions, see `resources/notation_guide.md`. For further reading on any topic, see `resources/bibliography.md`.*

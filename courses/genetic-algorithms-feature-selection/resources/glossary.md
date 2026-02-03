# Glossary: Genetic Algorithms for Feature Selection

## A

**Adaptive Operator**
: A genetic operator (crossover, mutation, selection) that adjusts its behavior based on population state, diversity, or convergence metrics. Example: increasing mutation rate when diversity is low.

**Allele**
: A specific value that a gene can take. In binary encoding for feature selection, alleles are 0 (not selected) or 1 (selected).

## B

**Binary Encoding**
: Chromosome representation using binary strings (0s and 1s). For feature selection, each bit represents whether a feature is selected (1) or not (0).

**Bit-Flip Mutation**
: A mutation operator that flips each bit with probability $p_m$. For binary chromosomes, changes 0→1 or 1→0.

## C

**Chromosome**
: The genetic representation of a solution. In feature selection, typically a binary vector where each position represents a feature.

**Convergence**
: The state where the population becomes homogeneous and no longer produces improved solutions. Can indicate finding the optimum or premature convergence to a local optimum.

**Crossover (Recombination)**
: A genetic operator that combines genetic material from two parent chromosomes to create offspring. Common types: single-point, two-point, uniform.

**Crowding Distance**
: A metric used in NSGA-II to maintain diversity along the Pareto front. Measures how isolated a solution is from its neighbors in objective space.

**Cross-Validation**
: A model validation technique that partitions data into training and validation sets. Essential for reliable fitness evaluation in wrapper methods.

**Curse of Dimensionality**
: The phenomenon where model performance degrades as the number of features increases relative to the number of samples. Motivates feature selection.

## D

**DEAP (Distributed Evolutionary Algorithms in Python)**
: A Python framework for implementing evolutionary algorithms, including genetic algorithms, genetic programming, and evolution strategies.

**Diversity**
: The variety of solutions in the population. High diversity enables exploration; low diversity may indicate convergence.

**Dominance (Pareto)**
: Solution A dominates solution B if A is at least as good on all objectives and strictly better on at least one. Non-dominated solutions form the Pareto front.

## E

**Elitism**
: A strategy that preserves the best individuals across generations, ensuring the best solution never degrades.

**Embedded Method**
: Feature selection performed during model training (e.g., L1 regularization, tree-based importance). Contrasts with filter and wrapper methods.

**Evolution**
: The iterative process of applying selection, crossover, and mutation to improve population fitness over generations.

**Exploitation**
: Searching near known good solutions to refine them. Balanced with exploration to avoid local optima.

**Exploration**
: Searching diverse regions of the solution space to discover new promising areas. Balanced with exploitation for effective search.

## F

**Feature**
: An input variable or predictor used in a model. Feature selection identifies the most relevant subset.

**Filter Method**
: Feature selection using statistical measures (correlation, mutual information) independent of the predictive model.

**Fitness Function**
: The objective function that evaluates how good a solution is. For feature selection, typically combines prediction accuracy and parsimony.

**Fitness Landscape**
: A visualization of fitness values across the solution space. Rugged landscapes have many local optima; smooth landscapes are easier to optimize.

**Forward Selection**
: A wrapper method that starts with no features and iteratively adds the most beneficial feature at each step.

## G

**Gene**
: A single position in the chromosome. In feature selection, each gene corresponds to one feature.

**Generation**
: One iteration of the evolutionary process: evaluation, selection, crossover, mutation, replacement.

**Genetic Algorithm (GA)**
: A metaheuristic optimization algorithm inspired by natural evolution, using selection, crossover, and mutation to search for optimal solutions.

**Genetic Operator**
: Operations that create new solutions from existing ones: selection, crossover, mutation.

**Genotype**
: The chromosome representation (e.g., binary string [1,0,1,0,1]). Contrasts with phenotype (the actual solution).

**Global Optimum**
: The best possible solution across the entire search space. GAs seek global optima but may settle for good local optima.

## H

**Hall of Fame**
: A data structure in DEAP that preserves the best individuals found during evolution, regardless of whether they survive selection.

**Hyperparameter**
: A parameter that controls GA behavior but is not evolved (e.g., population size, mutation rate, crossover probability).

**Hypervolume**
: A quality indicator for multi-objective optimization measuring the volume of objective space dominated by a set of solutions.

## I

**Individual**
: A single candidate solution in the population, represented by a chromosome.

**Informative Feature**
: A feature that provides useful information for predicting the target variable. Contrasts with irrelevant or redundant features.

## L

**L1 Regularization (Lasso)**
: A penalty term $\lambda \sum |\beta_i|$ that drives some coefficients to exactly zero, performing embedded feature selection.

**Lag Feature**
: In time series, a feature created by shifting a variable backward in time (e.g., $y_{t-1}, y_{t-2}$).

**Local Optimum**
: A solution that is better than all nearby solutions but may not be the global optimum.

## M

**Memetic Algorithm**
: A hybrid GA that combines evolutionary search with local search (e.g., hill climbing) applied to individuals.

**Multi-Objective Optimization**
: Optimizing multiple conflicting objectives simultaneously (e.g., minimize error AND minimize features). Produces a Pareto front of tradeoff solutions.

**Mutation**
: A genetic operator that introduces random changes to chromosomes, maintaining diversity and enabling exploration.

**Mutation Rate ($p_m$)**
: The probability of mutating each gene. Typical values: 1/n to 0.05 for binary encoding.

**Mutual Information (MI)**
: A filter method metric measuring the statistical dependency between a feature and the target: $I(X;Y) = H(Y) - H(Y|X)$.

## N

**Nested Cross-Validation**
: A CV strategy with inner loop for model selection and outer loop for performance estimation. Prevents overfitting the validation set in wrapper methods.

**NSGA-II**
: Non-dominated Sorting Genetic Algorithm II, a popular multi-objective GA using non-dominated sorting and crowding distance.

## O

**Objective Function**
: Same as fitness function. The function to optimize (minimize or maximize).

**Offspring**
: New individuals created through crossover and mutation of parent individuals.

**Overfitting**
: When a model learns noise instead of signal, performing well on training data but poorly on test data. Feature selection helps prevent overfitting.

## P

**Pareto Front (Pareto Optimal Set)**
: The set of non-dominated solutions in multi-objective optimization. Each solution represents a different tradeoff between objectives.

**Parsimony**
: Preference for simpler solutions (fewer features). Often implemented as a penalty term in the fitness function.

**Phenotype**
: The actual solution represented by the genotype. For feature selection, the subset of selected features.

**Population**
: The set of candidate solutions (individuals) maintained by the GA across generations.

**Population Size**
: Number of individuals in the population. Larger populations explore more but increase computational cost. Typical: 50-200.

**Premature Convergence**
: When the population converges to a local optimum before finding the global optimum, often due to loss of diversity.

## R

**Rank Selection**
: A selection operator that assigns selection probability based on fitness rank rather than absolute fitness values.

**Recombination**
: See Crossover.

**Redundant Feature**
: A feature that provides information already captured by other features. Redundant features waste model capacity.

**Replacement Strategy**
: How offspring replace parents in the population. Generational: replace entire population. Steady-state: replace one or few individuals.

**Roulette Wheel Selection**
: Selection probability proportional to fitness: $P(i) = \text{fitness}_i / \sum \text{fitness}_j$.

## S

**Schema**
: A pattern in the chromosome (e.g., [1,*,0,*,1] where * is wildcard). Schema theorem explains why GAs work.

**Search Space**
: The set of all possible solutions. For binary feature selection with $p$ features: $2^p$ solutions.

**Selection**
: A genetic operator that chooses individuals to be parents based on fitness. Higher fitness → higher selection probability.

**Selection Pressure**
: How strongly selection favors high-fitness individuals. High pressure accelerates convergence but may reduce diversity.

**Single-Point Crossover**
: Crossover that selects one random point and swaps chromosome segments between parents.

**Stagnation**
: A state where fitness no longer improves across generations, often indicating convergence or insufficient diversity.

**Stationarity**
: A time series property where statistical properties (mean, variance) don't change over time. Important for valid time series modeling.

## T

**Time Series Split**
: A cross-validation strategy that respects temporal ordering, always training on past data and testing on future data.

**Tournament Selection**
: Randomly select $k$ individuals and choose the best. Larger $k$ → higher selection pressure.

**Two-Point Crossover**
: Crossover that selects two random points and swaps the middle segment between parents.

## U

**Uniform Crossover**
: Crossover that independently decides for each gene whether to inherit from parent 1 or parent 2 (e.g., based on a random mask).

## V

**Validation Set**
: Data held out from training to evaluate model performance. In wrapper methods, used to evaluate feature subset quality.

## W

**Walk-Forward Validation**
: A time series CV strategy that simulates real forecasting: train on historical data, test on next period, then expand training window.

**Wrapper Method**
: Feature selection that uses a predictive model to evaluate feature subsets. GAs are wrapper methods.

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $p$ | Number of features |
| $n$ | Number of samples |
| $X$ | Feature matrix ($n \times p$) |
| $y$ | Target variable ($n \times 1$) |
| $s$ | Binary selection vector ($p \times 1$) |
| $s_i$ | Selection of feature $i$ (0 or 1) |
| $X_s$ | Feature matrix with selected features only |
| $\|\|s\|\|_0$ | L0 norm: count of non-zero elements (number of selected features) |
| $f(s)$ | Fitness function |
| $p_c$ | Crossover probability |
| $p_m$ | Mutation probability |
| $\lambda$ | Parsimony penalty weight |
| $\mathcal{P}$ | Pareto optimal set |
| $I(X;Y)$ | Mutual information between X and Y |
| $H(X)$ | Entropy of X |

## Acronyms

| Acronym | Full Name |
|---------|-----------|
| GA | Genetic Algorithm |
| DEAP | Distributed Evolutionary Algorithms in Python |
| EA | Evolutionary Algorithm |
| NSGA-II | Non-dominated Sorting Genetic Algorithm II |
| CV | Cross-Validation |
| MSE | Mean Squared Error |
| MAE | Mean Absolute Error |
| MI | Mutual Information |
| RFE | Recursive Feature Elimination |
| FCBF | Fast Correlation-Based Filter |
| mRMR | minimum Redundancy Maximum Relevance |
| PSO | Particle Swarm Optimization |
| SA | Simulated Annealing |
| ACO | Ant Colony Optimization |
| GP | Genetic Programming |
| ES | Evolution Strategy |

## Common Parameter Ranges

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Population size | 50-200 | Larger for complex problems |
| Generations | 30-100 | Until convergence |
| Crossover probability | 0.6-0.9 | Usually high |
| Mutation probability | 1/p to 0.1 | Lower for binary encoding |
| Tournament size | 2-7 | Higher = more selection pressure |
| Elitism count | 1-5 | Preserve best individuals |
| Parsimony weight | 0.001-0.1 | Balance accuracy vs. simplicity |

## References

See course bibliography for detailed references on all terms.

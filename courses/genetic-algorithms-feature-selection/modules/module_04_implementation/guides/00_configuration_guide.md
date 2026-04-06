# Choosing Your GA Configuration

> **Reading time:** ~12 min | **Module:** 4 — Implementation | **Prerequisites:** Modules 1-3

## In Brief

Before writing a single line of GA code, you face a series of interconnected design decisions: how to encode solutions, how to select parents, how to combine and perturb them, when to stop, and how to validate results. Each decision shapes the search behavior. Getting one wrong can waste hours of computation on a GA that converges prematurely, explores aimlessly, or overfits the training set. This guide walks through every decision a practitioner must make, with tradeoffs and recommended defaults for feature selection problems.

<div class="callout-key">

**Key Concept Summary:** A GA's performance is determined more by its configuration than by the cleverness of any single operator. The six critical decisions -- encoding, selection, crossover, mutation rate, population size, and stopping criteria -- interact as a system. A high mutation rate compensates for small populations. Strong selection pressure demands higher diversity mechanisms. The defaults in this guide are calibrated for feature selection on tabular datasets with 20-500 features.

</div>

## Decision 1: Encoding Type

**What it controls:** How a candidate solution (feature subset) is represented as a chromosome.

**Options:**

- **Binary encoding** -- Each gene is 0 (exclude) or 1 (include). The chromosome has length equal to the number of features.
- **Integer encoding** -- Each gene holds a feature index. The chromosome has a fixed length equal to the desired number of features.
- **Variable-length encoding** -- Chromosome length varies; each gene is a feature index. The GA can explore different subset sizes naturally.

**Tradeoffs:**

Binary encoding is the standard for feature selection. It is simple, works with all standard operators (uniform crossover, bit-flip mutation), and lets the GA discover the optimal subset size automatically. Integer encoding forces you to pre-specify how many features to select, which removes a degree of freedom. Variable-length encoding is flexible but requires specialized operators and complicates crossover.

**Recommended default:** Binary encoding. Every feature gets a bit. This is what DEAP, sklearn, and most GA libraries assume.

**When to deviate:** Use integer encoding when you have a hard budget constraint (e.g., "select exactly 10 features") and want to enforce it structurally rather than through penalties. Use variable-length only if you have extensive experience with GA operator design.

## Decision 2: Selection Operator

**What it controls:** How parents are chosen for reproduction. This determines *selection pressure* -- how strongly the GA favors fit individuals over weak ones.

**Options:**

- **Tournament selection** -- Pick `k` random individuals, select the best. Higher `k` = higher pressure.
- **Roulette wheel (fitness-proportional)** -- Probability of selection proportional to fitness. Sensitive to fitness scaling.
- **Rank-based selection** -- Probability based on rank, not raw fitness. Robust to fitness scaling.
- **Truncation selection** -- Select the top `p%` of the population. Maximum pressure.

**Tradeoffs:**

Tournament selection is the workhorse of modern GAs because it has a single tunable parameter (`k`), is insensitive to fitness scaling, and works for both minimization and maximization without modification. Roulette wheel can cause problems when fitness values are similar (everyone has equal chance) or when one individual dominates (premature convergence). Rank-based is robust but slower computationally. Truncation is simple but aggressive -- it discards diversity fast.

**Recommended default:** Tournament selection with `k=3`. This gives moderate selection pressure: the best individual in a random group of three becomes a parent.

**When to deviate:** Use `k=2` (lower pressure) for small populations or highly rugged landscapes where you need more exploration. Use `k=5-7` when you have large populations (200+) and want faster convergence. Avoid roulette wheel unless you have a strong reason.

## Decision 3: Crossover Type

**What it controls:** How two parent chromosomes are combined to produce offspring.

**Options:**

- **Uniform crossover** -- Each bit independently inherited from parent 1 or parent 2 with probability 0.5. No positional bias.
- **Single-point crossover** -- Pick a random point, swap everything after it. Creates positional bias.
- **Two-point crossover** -- Pick two random points, swap the segment between them.
- **Group-aware crossover** -- Swap entire feature groups (e.g., all one-hot columns) as units.

**Tradeoffs:**

Uniform crossover is strongly preferred for feature selection because features are *unordered*. With single-point crossover, features near the beginning of the chromosome are less likely to be separated than features near the end, creating an artificial positional dependency that does not exist in the problem. Uniform crossover treats every feature independently, which matches the structure of feature selection. Group-aware crossover is needed when you have one-hot encoded categoricals or other feature groups that must be swapped as units to maintain validity.

**Recommended default:** Uniform crossover with `indpb=0.5` (each bit has 50% chance of coming from each parent).

**When to deviate:** Use group-aware crossover when your feature set includes one-hot encoded categories, polynomial feature groups, or hierarchical features with dependencies. Single-point and two-point crossover are appropriate for problems with positional structure (e.g., time series lag sequences) but not for general feature selection.

## Decision 4: Mutation Rate

**What it controls:** The probability that each bit (gene) in a chromosome is flipped after crossover. This is the primary exploration mechanism.

**Tradeoffs:**

Too low and the GA cannot escape local optima -- it relies entirely on crossover to generate novelty, which is insufficient for fine-grained exploration. Too high and the GA degenerates into random search, destroying good solutions faster than crossover can assemble them. The sweet spot depends on chromosome length: longer chromosomes need lower per-gene rates to avoid excessive disruption.

**Recommended default:** `1/n_features`. For 100 features, use 0.01. For 50 features, use 0.02. This heuristic ensures that, on average, one feature flips per individual per generation -- enough to explore without being destructive.

**When to deviate:** Use higher rates (2-3x the default) in early generations if you want aggressive exploration, then decay. Use lower rates (0.5x the default) if your fitness function is expensive and you cannot afford wasted evaluations on heavily mutated individuals. Consider adaptive mutation (Module 5.3) to avoid choosing a fixed rate altogether.

<div class="callout-insight">

The `1/n_features` heuristic comes from the intuition that flipping exactly one feature per individual creates a minimal perturbation -- the smallest possible step in the binary search space. This is analogous to the learning rate in gradient descent: you want steps large enough to make progress but small enough to not overshoot good solutions.

</div>

## Decision 5: Population Size

**What it controls:** The number of candidate solutions maintained and evolved each generation. This is the primary diversity mechanism.

**Tradeoffs:**

Larger populations maintain more diversity, explore more of the search space, and are less likely to converge prematurely. But each generation requires evaluating every individual's fitness, which is the computational bottleneck. Doubling population size doubles the cost per generation. Smaller populations converge faster (fewer evaluations per generation) but are more likely to get stuck.

**Recommended default:** 2-5x `n_features`. For 50 features, use 100-250 individuals. For 100 features, use 200-500.

The reasoning: the search space for feature selection is $2^n$, which grows exponentially. The population needs to be large enough to maintain representatives from different regions of this space. The `2-5x` rule is a practitioner heuristic that balances diversity against computational cost for typical tabular datasets.

**When to deviate:** Use smaller populations (1-2x `n_features`) when fitness evaluation is expensive (e.g., training a neural network per evaluation) and you need to minimize total evaluations. Use larger populations (5-10x) when you have cheap fitness functions, many features (500+), or multi-objective optimization where you need to approximate the Pareto front.

<div class="callout-warning">

A population of 20 for a 200-feature problem is almost certainly too small. The GA will converge prematurely to whatever good solution it finds first, missing potentially better regions of the search space. If you cannot afford a larger population, consider using surrogate fitness models (Module 5.1) to reduce per-evaluation cost.

</div>

## Decision 6: Crossover Probability

**What it controls:** The probability that a selected pair of parents undergoes crossover (vs. passing through unchanged).

**Tradeoffs:**

Crossover is the primary *exploitation* operator -- it combines good building blocks from different parents. A high crossover probability means most offspring are recombinations. A low probability means most offspring are clones of a single parent, with only mutation providing variation. If crossover probability is too low, the GA wastes selection effort by not combining the parents it carefully chose.

**Recommended default:** 0.7-0.9. Most practitioners use 0.8.

**When to deviate:** Lower crossover probability (0.5-0.6) if you observe that crossover is disrupting good solutions -- this can happen when the population has already converged and parents are similar, so crossover just shuffles equivalent genes. Raise crossover probability (0.9-1.0) in early search when diversity is high and combining different solutions is productive.

## Decision 7: Stopping Criteria

**What it controls:** When the GA stops evolving and returns its best solution.

**Options:**

- **Fixed generations** -- Stop after `N` generations regardless of progress.
- **Early stopping** -- Stop if no improvement for `k` consecutive generations.
- **Fitness threshold** -- Stop when fitness reaches a target value.
- **Time limit** -- Stop after a wall-clock time budget.
- **Evaluation budget** -- Stop after a total number of fitness evaluations.

**Tradeoffs:**

Fixed generations is the simplest but wasteful -- the GA may converge at generation 20 and waste 80 more generations, or it may still be improving at generation 100. Early stopping is the most practical default: it adapts to the problem's convergence speed. Fitness thresholds require domain knowledge about what "good enough" looks like. Time limits and evaluation budgets are useful in production settings where compute cost matters.

**Recommended default:** Early stopping with patience of 10-15 generations, combined with a maximum generation cap of 50-100.

**When to deviate:** Use longer patience (20-30) for high-dimensional problems where the GA makes slow but steady progress. Use shorter patience (5-10) for fast fitness functions where you want quick iteration. Always combine early stopping with a hard maximum to prevent runaway execution.

## Decision 8: Validation Strategy

**What it controls:** How the fitness function estimates the quality of a feature subset.

**Options:**

- **k-fold cross-validation** -- Standard for i.i.d. data. 5-fold is the default.
- **Time series split** -- Required for temporal data. Expanding or rolling window.
- **Nested cross-validation** -- Outer loop for feature selection evaluation, inner loop for model training.
- **Hold-out validation** -- Split once, evaluate on held-out set. Fast but noisy.

**Tradeoffs:**

k-fold is the standard but is biased (uses the same data for both training and validation of the same individual). Nested CV removes this bias but is 5-10x more expensive. Time series split is mandatory for temporal data -- using k-fold on time series causes information leakage from future data, producing fitness estimates that are unrealistically optimistic. Hold-out is the cheapest but has high variance, especially for small datasets.

**Recommended default:** 5-fold CV for tabular data; `TimeSeriesSplit` with 5 splits for temporal data.

**When to deviate:** Use 3-fold CV when fitness evaluation is expensive and you need to reduce cost per evaluation. Use nested CV when you need unbiased performance estimates (e.g., for publication or regulatory requirements). Use hold-out only during rapid prototyping.

<div class="callout-warning">

The fitness function is the single most expensive component of a GA -- it runs once per individual per generation. With population 100 and 50 generations, that is 5000 evaluations. Each evaluation involves a full cross-validation loop. Reducing CV folds from 5 to 3 cuts total compute by 40%.

</div>

## Summary Decision Table

| Decision | Recommended Default | Heuristic / Reasoning | When to Deviate |
|---|---|---|---|
| **Encoding** | Binary (0/1 per feature) | Simplest, works with all operators | Integer if fixed budget; variable-length for advanced users |
| **Selection** | Tournament, k=3 | Moderate pressure, insensitive to scaling | k=2 for small pops; k=5-7 for large pops |
| **Crossover** | Uniform, indpb=0.5 | Features are unordered; no positional bias | Group-aware for one-hot/hierarchical features |
| **Mutation rate** | 1/n_features | One flip per individual on average | Higher for early exploration; adaptive for auto-tuning |
| **Population size** | 2-5x n_features | Diversity proportional to search space | Smaller if fitness is expensive; larger for multi-objective |
| **Crossover prob** | 0.8 | Crossover is the primary search operator | Lower if population is already converged |
| **Stopping** | Early stopping, patience=10-15 | Adapts to convergence speed | Longer patience for high-dimensional; time limits for production |
| **Validation** | 5-fold CV (or TimeSeriesSplit) | Standard unbiased estimate | 3-fold if expensive; nested CV if unbiased estimate needed |

<div class="callout-insight">

These defaults are not magic numbers -- they are starting points refined through decades of GA research and practice. The key insight is that GA parameters interact: if you increase selection pressure (larger tournament), you need more diversity (larger population or higher mutation) to compensate. Changing one parameter in isolation can degrade performance. When in doubt, run a small pilot with the defaults above and adjust based on convergence plots.

</div>

## Practice Problems

1. **Configuration Design**
   You have 200 features, a Random Forest fitness function that takes 2 seconds per evaluation, and a 1-hour time budget. Design a complete GA configuration: population size, generations, crossover probability, mutation rate, stopping criteria. How many total evaluations can you afford? Does your configuration fit within budget?

2. **Tradeoff Analysis**
   You run a GA with population 50 on a 100-feature problem and it converges by generation 15 to a suboptimal solution. Diagnose the problem. Which parameter(s) would you change, and in which direction? Explain the tradeoff involved.

3. **Conceptual: Why Uniform Crossover?**
   Explain in your own words why uniform crossover is preferred over single-point crossover for feature selection. What would happen if you used single-point crossover on a binary-encoded feature selection chromosome where the first 10 features are noise and the last 10 are informative?

4. **Validation Strategy Choice**
   You are selecting features for a stock return prediction model using daily data from 2015-2024. Which validation strategy do you use? What happens if you accidentally use standard 5-fold CV instead?

---

**Next:** [DEAP Implementation](./01_deap_implementation.md) | [Custom Operators](./02_custom_operators.md)

# Common Misconceptions About Genetic Algorithms

> **Reading time:** ~8 min | **Module:** 0 — Foundations | **Prerequisites:** 01 Feature Selection Problem, 02 Optimization Basics

## In Brief

GAs are powerful but widely misunderstood. This guide addresses the five most common misconceptions practitioners hold about genetic algorithms for feature selection. Each misconception is paired with the reality, why it matters, and a concrete example.

<div class="callout-key">

<strong>Key Concept:</strong> Genetic algorithms are stochastic satisficing optimizers -- they find good solutions quickly, not perfect solutions guaranteed. Understanding what GAs actually promise (and what they do not) prevents wasted effort, miscalibrated expectations, and flawed experimental designs.

</div>

## Misconception 1: GAs Always Find the Global Optimum

### The Myth

"If I run the GA long enough, it will find the best possible feature subset."

### The Reality

GAs are **heuristic** optimizers. They have no guarantee of finding the global optimum. The No Free Lunch Theorem (Module 0, Guide 02) formalizes this: no search algorithm is universally optimal. GAs explore the search space efficiently, but they can and do get trapped in local optima, converge prematurely, or miss rare high-quality solutions.

What GAs actually deliver is a **high-quality solution found in reasonable time**. For feature selection with 50 features ($2^{50}$ possible subsets), finding the provably best subset would require exhaustive evaluation. A GA typically finds a subset within a few percent of optimal in minutes rather than millennia.

### Why It Matters

If you believe GAs find the global optimum, you will:
- Run the GA once and trust the result completely
- Skip robustness checks (multiple runs, different seeds)
- Over-interpret small differences between GA results

### Example

A GA selecting from 50 commodity forecasting features finds a 7-feature subset with 67% accuracy. Running the GA 10 more times with different random seeds produces subsets with accuracies ranging from 64% to 68%. The "best" run is not guaranteed to be globally optimal -- it is the best of 11 stochastic samples. The correct approach: run multiple times, report the range, and use the consensus features (those appearing in most runs).

<div class="callout-warning">

<strong>Warning:</strong> A single GA run is a single sample from a stochastic process. Always run the GA multiple times (at least 5-10 runs) with different random seeds and report the distribution of results, not just the best run.

</div>

---

## Misconception 2: More Generations Always Produce Better Results

### The Myth

"My GA hasn't converged yet after 100 generations. If I run it for 500, the result will be much better."

### The Reality

GA improvement follows a curve of **diminishing returns**. Most progress happens in the first 20-30% of generations. After that, the population converges and additional generations produce smaller and smaller improvements. Worse, running too many generations can cause **overfitting to the fitness function**: the GA finds feature subsets that exploit quirks in the training/validation split rather than genuine predictive signal.

The relationship between generations and quality looks roughly like this:

```
Quality
   │           ┌───────────────── Overfitting risk zone
   │          ╱
   │      ●──●──●──●──●──●──●──●──●   ← Diminishing returns
   │    ●
   │  ●
   │ ●
   │●
   │
   └────────────────────────────────→
    10   50  100  200  300  400  500
                  Generations
```

### Why It Matters

Running excessive generations wastes computational budget that could be better spent on:
- More independent runs (better coverage of the search space)
- Larger population (more diversity per generation)
- More robust fitness evaluation (more CV folds, walk-forward windows)

### Example

On the commodity dataset, a GA with population 100 achieves 65% accuracy at generation 50, 67% at generation 100, 67.3% at generation 200, and 67.4% at generation 500. The last 400 generations bought 0.1% accuracy at 4x the cost. Running 5 independent 100-generation runs would have been more informative and cheaper.

<div class="callout-insight">

<strong>Rule of thumb:</strong> If the best fitness has not improved for 20-30 consecutive generations, the GA has effectively converged. Stop early and invest the saved compute elsewhere.

</div>

---

## Misconception 3: Larger Population Is Always Better

### The Myth

"I'll set population size to 1,000 to make sure the GA explores thoroughly."

### The Reality

Population size involves a **fundamental tradeoff** between exploration breadth and computational cost. Each individual in the population requires a fitness evaluation, which for feature selection means training and cross-validating a model. Doubling the population doubles the cost per generation.

A population that is too small (< 20) lacks diversity and converges prematurely. A population that is too large (> 500 for typical feature selection) wastes evaluations on redundant exploration -- you get diminishing returns because many individuals are exploring similar regions.

| Population Size | Diversity | Cost per Generation | Convergence | Risk |
|----------------|-----------|-------------------|-------------|------|
| 20 | Low | Low | Fast (often premature) | Misses good regions |
| 50-100 | Moderate | Moderate | Balanced | Good default |
| 200-500 | High | High | Slow but thorough | Expensive |
| 1,000+ | Very high | Very high | Very slow | Wastes compute |

### Why It Matters

The standard heuristic is **population size = 2 to 5 times the number of features**. For 50 features, that means 100-250 individuals. Going much beyond that rarely improves results but always increases cost.

### Example

For the 50-feature commodity dataset with fitness evaluation taking 0.5 seconds per individual:

- Population 100, 100 generations: 100 x 100 x 0.5s = **83 minutes**, finds 67% accuracy
- Population 500, 100 generations: 500 x 100 x 0.5s = **7 hours**, finds 67.5% accuracy
- Population 100, 100 generations, 5 runs: 5 x 83 min = **7 hours**, finds 64-68% range with consensus features

The same 7-hour budget is better spent on multiple runs than a single large-population run.

---

## Misconception 4: Mutation Rate Does Not Matter Much

### The Myth

"Mutation is just random noise. Crossover does the real work. I'll set mutation rate to something small and forget about it."

### The Reality

Mutation is **the only operator that introduces genuinely new genetic material** into the population. Crossover recombines what already exists; mutation creates what has never existed. Without adequate mutation, the population loses diversity over generations as selection removes weaker individuals. Eventually, all individuals become nearly identical (premature convergence), and crossover between identical parents produces identical offspring -- the search stalls completely.

The standard heuristic is **mutation rate = 1/n** where n is the chromosome length (number of features). For 50 features, that means a mutation rate of 0.02, which flips approximately one feature per individual per generation.

### Why It Matters

Setting mutation rate too low (< 0.001) causes premature convergence. Setting it too high (> 0.1) turns the GA into random search -- good solutions are destroyed faster than they can be exploited. The difference between a well-tuned and poorly-tuned mutation rate can be 5-10% in final accuracy.

### Example

Same 50-feature commodity problem, population 100, 100 generations:

| Mutation Rate | Diversity at Gen 50 | Best Accuracy | Behavior |
|--------------|--------------------:|-------------:|----------|
| 0.001 | 5% | 62% | Premature convergence by gen 20 |
| 0.02 (1/n) | 35% | 67% | Balanced exploration and exploitation |
| 0.1 | 85% | 59% | Near-random search, cannot exploit good solutions |
| 0.5 | 98% | 52% | Pure random search |

<div class="callout-danger">

<strong>Danger:</strong> A mutation rate of 0.5 means half of all feature selections flip every generation. This is not a genetic algorithm -- it is a random search wearing a GA costume. The algorithm retains no memory of good solutions.

</div>

---

## Misconception 5: Feature Selection Results Are Deterministic

### The Myth

"I ran the GA and it selected features 3, 7, 12, 19, and 31. Those are the right features."

### The Reality

GAs are **stochastic** algorithms. Every component involves randomness: initial population generation, parent selection (tournament sampling), crossover (swap mask), and mutation (bit flips). Running the same GA with a different random seed will produce a different result.

This is not a flaw -- it is a feature. The stochasticity allows exploration of diverse regions of the search space. But it means that any single run is one sample from a distribution of possible outcomes. The correct way to interpret GA results is statistically:

1. Run the GA N times (N >= 10) with different random seeds
2. Record which features are selected in each run
3. Report **feature selection frequency**: "Feature 7 was selected in 9 of 10 runs"
4. Use features that appear in most runs (e.g., > 70% of runs) as the robust set

### Why It Matters

If you treat a single GA run as deterministic truth, you will:
- Overfit to a specific random outcome
- Miss features that are important but happened not to appear in that run
- Include features that appeared by chance in that run

### Example

Ten runs on the commodity dataset produce these selection frequencies:

| Feature | Times Selected (out of 10) | Interpretation |
|---------|---------------------------:|----------------|
| 5-day return | 10 | Core feature -- always selected |
| RSI | 9 | Core feature |
| VIX | 8 | Core feature |
| MACD | 7 | Likely important |
| Rig count | 4 | Marginal -- may or may not help |
| Day-of-week | 1 | Noise -- selected by chance |

The robust feature set is {5-day return, RSI, VIX, MACD}. Day-of-week appeared in one run purely by chance. A practitioner who ran only that one run might have included it.

<div class="callout-insight">

<strong>Best practice:</strong> Report feature selection frequency across multiple GA runs, not the result of a single run. Features selected in > 70% of runs are robust; features selected in < 30% are likely noise.

</div>

---

## Summary: What GAs Actually Promise

| GAs Promise | GAs Do Not Promise |
|------------|-------------------|
| Good solutions found quickly | The globally optimal solution |
| Handling of feature interactions | Deterministic, reproducible results (without seed fixing) |
| Escape from local optima (probabilistic) | Guaranteed escape from every local optimum |
| Scalability to hundreds of features | That more compute always means better results |
| A flexible framework for diverse problems | That default parameters work for every problem |

## Practice Problems

### Problem 1: Convergence Diagnosis

**Question:** You run a GA for 200 generations. The best fitness improves rapidly from generation 1-30, then barely changes from generation 30-200. The average population fitness, however, keeps climbing toward the best fitness. What is happening, and what would you do differently?

**Hint:** Think about the relationship between population diversity and convergence.

### Problem 2: Robustness Assessment

**Question:** You run a GA 10 times on the same dataset. In 8 runs, features {A, B, C, D} are always selected. In 2 runs, the GA selects completely different feature sets with similar fitness. What does this tell you about the fitness landscape? What action would you take?

### Problem 3: Conceptual Comparison

**Question:** Explain why increasing population size from 100 to 1,000 is not equivalent to running 10 independent runs of population size 100, even though both involve 1,000 individuals. What does each approach explore that the other might miss?

---

**Next:** [Companion Slides](./01_feature_selection_challenge_slides.md) | [Module 1: GA Fundamentals](../../module_01_ga_fundamentals/guides/01_encoding.md)

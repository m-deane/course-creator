---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: Welcome to Module 5. Set the stage: students have seen filter methods (Module 1), information theory (Module 2), wrappers (Module 3), and embedded methods (Module 4). GAs are a different paradigm — population-based, stochastic, and able to handle complex feature interactions. Emphasise that this is about evolving solutions, not computing a score. -->

# Genetic Algorithms for Feature Selection
## Module 5 — Fundamentals

**Population-based search across the $2^p$ binary feature space**

---

<!-- Speaker notes: Ground the motivation. The key insight is that all methods studied so far make local decisions (add/remove one feature at a time, or score features independently). GAs make global decisions — they search combinations. Draw the analogy to optimisation: gradient descent vs evolutionary search. -->

## Why Genetic Algorithms?

Filter methods score features **independently** — they miss interactions.

Wrappers are **greedy** — they get trapped in local optima.

GAs search **populations of complete subsets simultaneously**:

| Property | Filter | Wrapper | GA |
|:---|:---:|:---:|:---:|
| Handles interactions | No | Partial | Yes |
| Escapes local optima | N/A | Rarely | Yes |
| Scales to $p > 500$ | Yes | No | Moderate |
| Requires gradient | No | No | No |
| Search is exhaustive | No | No | No |

> GAs trade computational cost for global search capability.

---

<!-- Speaker notes: Introduce the chromosome encoding. Use the blackboard analogy: a chromosome is a row of light switches — each switch controls whether a feature is included. Show the concrete example. Stress that this encoding is the foundation of everything that follows. -->

## Binary Chromosome Encoding

Each **chromosome** is a binary string of length $p$ (number of features).

$$s \in \{0, 1\}^p, \quad s_i = \begin{cases} 1 & \text{feature } i \text{ selected} \\ 0 & \text{feature } i \text{ excluded} \end{cases}$$

**Example** with $p = 10$ features:

```
Feature:     F0  F1  F2  F3  F4  F5  F6  F7  F8  F9
Chromosome: [ 1   0   1   1   0   0   1   0   1   0 ]
                    ↑               ↑
             Selected features: {0, 2, 3, 6, 8}   →  5 features
```

- **Search space**: $2^{10} = 1{,}024$ subsets (for $p=10$)
- **Search space**: $2^{50} \approx 10^{15}$ subsets (for $p=50$)

The GA searches this space **without evaluating all $2^p$ subsets**.

---

<!-- Speaker notes: Walk through each initialisation strategy. Biased initialisation is particularly important in practice — filter scores are cheap, and seeding the population intelligently reduces wasted generations. Emphasise that all strategies must guarantee at least one feature per individual. -->

## Population Initialisation Strategies

<div class="columns">

**Random** — uniform $P(s_i=1) = 0.5$

```python
def init_random(pop_size, n_features):
    return [Individual.random(n_features)
            for _ in range(pop_size)]
```
Expected $p/2$ features per individual.

**Biased** — $P(s_i=1) \propto \text{filter score}_i$

```python
scores = mutual_info_classif(X, y)
probs = normalise(scores, [0.1, 0.9])
```
Faster convergence, less diversity.

</div>

**Guided** — seed with expert/prior subsets:
```python
def init_guided(pop_size, n_features, seed_subsets):
    pop = [encode(subset) for subset in seed_subsets]
    while len(pop) < pop_size:
        pop.append(Individual.random(n_features))
    return pop
```

Always guarantee $\sum_i s_i \geq 1$ (repair if all-zero).

---

<!-- Speaker notes: This is the central diagram of the module. Walk through each step slowly. Fitness evaluation is the expensive step — everything else is cheap. The key word is 'generational': each iteration replaces the whole population (or most of it). Contrast with steady-state GAs which replace one individual at a time. -->

## The Generational Cycle

```mermaid
flowchart TD
    A([Initial Population P₀]) --> B[Evaluate Fitness]
    B --> C[Select Parents]
    C --> D[Crossover → Offspring]
    D --> E[Mutate Offspring]
    E --> F[Evaluate Offspring Fitness]
    F --> G{Elitism}
    G --> H[New Population P_{t+1}]
    H --> I{Termination?}
    I -- No --> C
    I -- Yes --> J([Return Best Individual])
```

**Termination criteria**: max generations | fitness plateau | wall-clock budget

---

<!-- Speaker notes: Tournament selection is the default. Use the analogy of a sports tournament. The key parameter is k — small k means upsets are common (high diversity), large k means the best always wins (low diversity). Show the table and let learners pick k=3 as a safe default. -->

## Tournament Selection

Pick $k$ individuals at random; return the best. Repeat for each parent slot.

$$P(\text{best individual selected}) = 1 - \left(1 - \frac{1}{N}\right)^k \approx \frac{k}{N}$$

```python
def tournament_select(population, k=3):
    contestants = np.random.choice(len(population), k, replace=False)
    winner = max(contestants, key=lambda i: population[i].fitness)
    return population[winner].copy()
```

| $k$ | Selection pressure | Convergence | Diversity |
|:---:|:---:|:---:|:---:|
| 2 | Low | Slow | High |
| **3–5** | **Medium** | **Balanced** | **Moderate** |
| 10 | High | Fast | Low |
| $N$ | Maximum | Very fast | Minimal |

**Recommendation**: $k = 3$ for most feature selection problems.

---

<!-- Speaker notes: Roulette wheel is the classic but has a known flaw — if one individual dominates, it takes over. Mention sigma scaling as a fix. SUS is the improved version that eliminates sampling noise. The key formula for SUS: equally-spaced pointers. Use the spinning wheel analogy for roulette, ruler analogy for SUS. -->

## Roulette Wheel and SUS

**Roulette Wheel** — probability proportional to fitness:

$$P(\text{select } i) = \frac{f_i}{\sum_j f_j}$$

**Problem**: dominance by high-fitness individuals → premature convergence.

**Stochastic Universal Sampling (SUS)** — $M$ equally-spaced pointers:

$$\text{pointer}_j = r + \frac{j-1}{M}, \quad r \sim U\!\left[0, \frac{1}{M}\right]$$

<div class="columns">

Roulette wheel: high variance, one spin per parent.

SUS: zero spread — each individual selected within ±1 of expected count.

</div>

> **SUS is always preferred over roulette wheel** when you need $M$ parents.

---

<!-- Speaker notes: Rank selection is robust because it's immune to fitness scaling. Explain that raw fitness values don't matter — only rank. This prevents the "superindividual" problem. The linear ranking formula gives controllable selection pressure via eta_plus. A value of 1.5 is a common default. -->

## Rank-Based Selection

Sort by fitness; assign probability by **rank** (not raw value):

$$P(\text{rank } r) = \frac{1}{N}\left(\eta^+ - (\eta^+ - \eta^-)\frac{r-1}{N-1}\right)$$

where $\eta^+ \in (1, 2]$ is the selection bias, $\eta^- = 2 - \eta^+$.

```python
def rank_select(population, eta_plus=1.5):
    N = len(population)
    sorted_idx = np.argsort([ind.fitness for ind in population])
    ranks = np.arange(1, N + 1)
    eta_minus = 2.0 - eta_plus
    probs = (1/N) * (eta_minus + (eta_plus - eta_minus) * (ranks-1)/(N-1))
    probs /= probs.sum()
    idx = np.random.choice(N, p=probs)
    return population[sorted_idx[idx]].copy()
```

**Advantage**: selection pressure is **constant** regardless of fitness variance — prevents dominance collapse.

---

<!-- Speaker notes: Crossover is where "innovation" happens. The analogy is sexual reproduction — offspring inherit traits from both parents. For feature selection: a child inherits some features from parent 1 and some from parent 2. Walk through each type with the visual. Emphasise that single-point creates position-dependent linkage, uniform does not. -->

## Crossover Operators — Overview

```
Parent 1: [1 1 1 1 0 0 0 0]
Parent 2: [0 0 0 0 1 1 1 1]
```

**Single-point** (cut at position 4):
```
Child 1: [1 1 1 1 | 1 1 1 1]    Child 2: [0 0 0 0 | 0 0 0 0]
```

**Two-point** (cut at 2 and 6):
```
Child 1: [1 1 | 0 0 1 1 | 0 0]  Child 2: [0 0 | 1 1 0 0 | 1 1]
```

**Uniform** (random mask per gene):
```
Mask:    [1 0 1 0 1 0 1 0]
Child 1: [1 0 1 0 1 0 1 0]      Child 2: [0 1 0 1 0 1 0 1]
```

| Operator | Positional bias | Diversity | Preferred when |
|:---|:---:|:---:|:---|
| Single-point | High | Low | Features ordered by relevance |
| Two-point | Moderate | Moderate | Unknown structure |
| Uniform | None | **Highest** | **Default for feature selection** |

---

<!-- Speaker notes: Feature-group-aware crossover is the advanced concept. The key idea: if you know features form semantic groups (e.g., lag-1,2,3 of the same variable), you should treat them as a unit. Standard crossover might split lag-1 into child 1 and lag-2,3 into child 2, destroying the group's utility. Block crossover prevents this. -->

## Feature-Group-Aware Crossover

When features form **semantic groups**, standard crossover may split them:

```
Groups: [F0,F1,F2] [F3,F4] [F5,F6,F7,F8,F9]
         (lag feats)  (MA)    (momentum indicators)

Parent 1: [1 1 1 | 0 0 | 1 1 1 1 1]   ← all lag features
Parent 2: [0 0 0 | 1 1 | 0 0 0 0 0]   ← only MAs
```

**Block crossover** — swap entire groups atomically:

```python
def block_crossover(p1, p2, groups):
    c1, c2 = p1.chromosome.copy(), p2.chromosome.copy()
    for group in groups:
        if np.random.random() < 0.5:
            c1[group], c2[group] = p2.chromosome[group], p1.chromosome[group]
    return Individual(c1), Individual(c2)
```

**Use when**: features come from time-series lags, polynomial expansions, or domain-defined groups. Dramatically improves convergence on structured feature spaces.

---

<!-- Speaker notes: Mutation prevents premature convergence. Analogy: mutation in biology introduces novel genetic variation. For GAs: without mutation, the population can only explore combinations of genes already present in the initial population. With mutation, any bit can flip at any time. The 1/p rate is theoretically motivated. -->

## Mutation Operators

**Bit-flip mutation** — each gene flips independently:

$$s_i' = \begin{cases} 1 - s_i & \text{with probability } p_m \\ s_i & \text{with probability } 1-p_m \end{cases}$$

```python
def bit_flip_mutate(ind, p_m=None):
    if p_m is None:
        p_m = 1.0 / len(ind.chromosome)   # expected 1 flip
    mask = np.random.random(len(ind.chromosome)) < p_m
    ind.chromosome[mask] ^= 1             # XOR flip
    if ind.chromosome.sum() == 0:         # repair all-zero
        ind.chromosome[np.random.randint(len(ind.chromosome))] = 1
    ind.fitness = None
```

| Rate | Expected flips | Effect |
|:---:|:---:|:---|
| $1/p$ | 1 | Standard — balanced |
| $1/\sqrt{p}$ | $\sqrt{p}$ | More exploration |
| $0.01$ | $0.01p$ | Conservative (may stagnate) |
| $0.5$ | $p/2$ | Destructive (random search) |

---

<!-- Speaker notes: Adaptive mutation is important in practice. The intuition: early in the run, explore broadly (high p_m); late in the run, exploit what you've found (low p_m). The diversity-triggered variant is the most responsive — when diversity collapses, immediately boost mutation. Show the formula and ask: what happens at t=0 and t=T_max? -->

## Adaptive Mutation Rates

**Problem**: fixed $p_m$ is a compromise — too high destroys good solutions, too low allows stagnation.

**Generational decay**:
$$p_m(t) = p_{m,\text{init}} \cdot \left(1 - \frac{t}{T}\right) + p_{m,\text{final}}$$

**Diversity-triggered** (most responsive):
$$p_m(t) = \begin{cases} p_{m,\text{high}} & \text{if } D(t) < D_{\min} \\ p_{m,\text{low}} & \text{otherwise} \end{cases}$$

where $D(t)$ = mean pairwise Hamming distance of population.

**Feature-count-preserving mutation** (swap mutation):
```python
def swap_mutate(ind, n_swaps=1):
    selected = np.where(ind.chromosome == 1)[0]
    unselected = np.where(ind.chromosome == 0)[0]
    drop = np.random.choice(selected, n_swaps, replace=False)
    add = np.random.choice(unselected, n_swaps, replace=False)
    ind.chromosome[drop] = 0
    ind.chromosome[add] = 1
```

---

<!-- Speaker notes: Elitism is simple but critical. The "best individual seen" should never be lost. Analogy: an Olympic record should not be erased because a worse athlete wins the next race. The key tradeoff is between preserving good solutions (high elitism) and maintaining diversity (low elitism). The 2-5% recommendation is widely supported empirically. -->

## Elitism — Preserving Best Solutions

**Without elitism**: best individual can be lost to crossover/mutation.

**With elitism**: top $e$ individuals are copied directly to next generation.

```python
def apply_elitism(old_pop, new_pop, n_elite):
    elites = sorted(old_pop, key=lambda x: x.fitness,
                    reverse=True)[:n_elite]
    worst = sorted(new_pop, key=lambda x: x.fitness)[:n_elite]
    for i, elite in enumerate(elites):
        new_pop[new_pop.index(worst[i])] = elite.copy()
    return new_pop
```

| Elitism $e/N$ | Effect |
|:---:|:---|
| 0% | Best solution can be lost — risky |
| **1–5%** | **Balanced — standard recommendation** |
| 10–20% | Fast convergence, diversity risk |
| >50% | Effectively local search |

> **Rule**: always use elitism $e \geq 1$. It costs nothing and always helps.

---

<!-- Speaker notes: Bring it all together. This slide shows the complete cycle with all components labelled. Ask the class: at which step does the GA most resemble a random search? (initialisation and mutation). At which step does it most resemble a greedy search? (selection). The power is in their combination. -->

## Putting It All Together

```python
class GAFeatureSelector:
    def fit(self, X, y):
        # Initialise
        pop = [Individual.random(X.shape[1]) for _ in range(cfg.pop_size)]
        for ind in pop: ind.fitness = self.evaluate(ind, X, y)

        for gen in range(cfg.n_generations):
            next_gen = []

            # Elitism
            elites = sorted(pop, key=lambda x: x.fitness, reverse=True)
            next_gen.extend([e.copy() for e in elites[:cfg.n_elite]])

            # Selection → Crossover → Mutation
            while len(next_gen) < cfg.pop_size:
                p1 = tournament_select(pop, cfg.k)
                p2 = tournament_select(pop, cfg.k)
                c1, c2 = uniform_crossover(p1, p2)
                bit_flip_mutate(c1, p_m)
                bit_flip_mutate(c2, p_m)
                c1.fitness = self.evaluate(c1, X, y)
                c2.fitness = self.evaluate(c2, X, y)
                next_gen.extend([c1, c2])

            pop = next_gen[:cfg.pop_size]
```

---

<!-- Speaker notes: The fitness function is the bridge between the GA mechanics and the actual feature selection task. Reinforce that all the operators are generic — the fitness function is domain-specific. Preview Module 5.2 which goes deep on fitness design. -->

## Fitness Function (Preview)

The fitness function is what makes the GA **domain-specific**.

$$\text{fitness}(s) = \underbrace{\text{CV-accuracy}(M, X_s, y)}_{\text{predictive quality}} - \underbrace{\lambda \cdot \frac{|s|}{p}}_{\text{parsimony penalty}}$$

- $M$ is the base model (Logistic Regression, Random Forest, ...)
- $X_s$ is the training data restricted to selected features
- $\lambda$ controls the accuracy/complexity tradeoff
- Caching by chromosome bytes key is **essential** (same subset reappears often)

**Edge cases** (covered in Guide 02):
- All-zero chromosome → fitness = -1 (immediate penalty)
- All-ones chromosome → fitness = baseline (valid, expensive)
- Degenerate population → diversity-triggered mutation

---

<!-- Speaker notes: Operator selection guide. This is the take-home message for practitioners. Students should walk away being able to configure a GA for their specific problem. Highlight that "uniform + tournament(3) + 1/p" is the safe default for most feature selection problems. -->

## Operator Selection Guide

| Scenario | Crossover | Selection | $p_m$ |
|:---|:---:|:---:|:---:|
| General ($p < 50$) | Uniform | Tournament $k=3$ | $1/p$ |
| High-dim ($p > 200$) | Two-point | Rank ($\eta^+=1.5$) | Adaptive |
| Feature groups exist | Block | Tournament $k=3$ | $1/p$ |
| Fast convergence | Uniform | Tournament $k=7$ | $1/(2p)$ |
| Exploration priority | Uniform | Roulette/SUS | $3/p$ |
| Fixed-size subset | N/A + swap mutation | Any | Swap only |

**Safe defaults** for feature selection:
```
crossover   = uniform,     p_c = 0.8
selection   = tournament,  k = 3
mutation    = bit-flip,    p_m = 1/p
elitism     = 2 individuals
pop_size    = 50–100
generations = 50–200
```

---

<!-- Speaker notes: Performance tips before heading to the notebooks. The cache is the single biggest win — mention that without caching, a typical run evaluates 50*100=5000 chromosomes, but only ~500 are unique. Parallel evaluation is covered in Notebook 02 (DEAP). Early stopping saves wall-clock time without sacrificing quality. -->

## Implementation Performance Tips

**1. Fitness caching** — same chromosome reappears frequently:
```python
cache: dict[bytes, float] = {}
key = ind.chromosome.tobytes()
if key in cache: return cache[key]
```

**2. Vectorised population evaluation**:
```python
# Evaluate all unevaluated individuals in one pass
unevaluated = [ind for ind in pop if ind.fitness is None]
```

**3. Early stopping** on fitness plateau:
```python
if stagnation >= patience:  # no improvement for N gens
    break
```

**4. Parallel evaluation** with `joblib` or DEAP (covered in Notebook 02):
```python
from joblib import Parallel, delayed
fitnesses = Parallel(n_jobs=-1)(
    delayed(evaluate)(ind, X, y) for ind in population
)
```

---

<!-- Speaker notes: Summarise the module and point forward. The key deliverables from this guide are: (1) ability to implement a GA from scratch, (2) ability to choose appropriate operators, (3) awareness of edge cases. Guides 02 and 03 go deeper on fitness and convergence respectively. -->

## Module 5.1 Summary

**Core concepts covered:**

- Binary chromosome encodes feature subset: gene $i = 1$ → feature $i$ selected
- Population initialisation: random, biased (filter scores), guided (domain knowledge)
- Generational cycle: evaluate → select → crossover → mutate → replace
- Selection: tournament (default), roulette wheel, rank, SUS
- Crossover: single-point, two-point, uniform, block (group-aware)
- Mutation: bit-flip ($p_m = 1/p$), adaptive rates, swap mutation (count-preserving)
- Elitism: preserve 1–5% of best individuals every generation
- Fitness caching: essential for computational efficiency

**Next:**
- **Guide 02**: Fitness function design — CV strategies, parsimony penalties, landscape analysis
- **Notebook 01**: Build the complete GA from scratch on real data

---

<!-- Speaker notes: Quick self-check before the notebook. These questions align with what students will implement in Notebook 01. They should be able to answer all five from memory after reading this guide. If not, point them back to the specific sections. -->

## Self-Check Questions

1. A chromosome has $p = 20$ genes. Gene 7 is 1 and gene 15 is 0. What does this mean for feature 7 and feature 15?

2. You have 50 individuals. You run tournament selection with $k = 10$. Is selection pressure high or low compared to $k = 2$? What is the risk?

3. Two parents are `[1,1,0,0,1,1,0,0]` and `[0,0,1,1,0,0,1,1]`. Crossover point is at position 4. Write out both children.

4. With $p = 30$ features, what is the standard mutation rate? How many bits do you expect to flip per mutation event?

5. You run a GA for 100 generations and notice the best fitness has not improved since generation 40. What are two possible causes and one remedy for each?

---

<!-- Speaker notes: References for deeper reading. Goldberg 1989 is the classic — worth skimming Chapter 1-3. Eiben & Smith is the modern textbook. The Real-coded GA paper is relevant if students want to explore continuous encodings. -->

## References

- **Goldberg, D.E. (1989)** — *Genetic Algorithms in Search, Optimization and Machine Learning* — Addison-Wesley. Classic foundational text.
- **Eiben, A.E. & Smith, J.E. (2015)** — *Introduction to Evolutionary Computing*, 2nd ed. — Springer. Modern comprehensive treatment.
- **Back, T. (1996)** — *Evolutionary Algorithms in Theory and Practice* — Oxford. Rigorous mathematical analysis of selection operators.
- **Deb, K. (2001)** — *Multi-Objective Optimization using Evolutionary Algorithms* — Wiley. Foundation for Guide 02 multi-objective fitness.
- **DEAP documentation** — https://deap.readthedocs.io — Framework used in Notebook 02.

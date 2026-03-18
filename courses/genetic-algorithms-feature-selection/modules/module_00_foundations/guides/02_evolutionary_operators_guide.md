# Evolutionary Operators: Selection, Crossover, and Mutation

## In Brief

A genetic algorithm (GA) is a population-based search method that maintains a collection of candidate solutions and iteratively improves them through three operators: selection, crossover, and mutation. Selection determines which individuals reproduce. Crossover combines genetic material from two parents to create offspring. Mutation introduces random variation to maintain diversity.

Together these three operators drive the GA through the search space, balancing exploitation of known good solutions with exploration of unknown regions.

## Key Insight

The power of a GA over greedy search comes from its population structure and the recombination mechanism. Crossover can combine complementary feature subsets from two different parents to produce a child that inherits the best of both. A greedy search commits to a single solution path; a GA explores many paths simultaneously and shares discoveries across the population.

## Formal Definition

Let $\mathcal{P}_t = \{x_1^{(t)}, \ldots, x_N^{(t)}\}$ be the population at generation $t$, where each individual $x_i^{(t)} \in \{0,1\}^p$ is a binary feature selection vector.

**Selection** produces a multiset of parents $\mathcal{P}'_t$ by sampling from $\mathcal{P}_t$ with probabilities derived from fitness.

**Crossover** with probability $p_c$ takes two parents $(p_1, p_2) \in \mathcal{P}'_t$ and produces two offspring $(c_1, c_2)$ by recombining their genes.

**Mutation** with probability $p_m$ flips each bit of an offspring independently.

**Replacement** forms $\mathcal{P}_{t+1}$ from the offspring (and possibly elite individuals from $\mathcal{P}_t$).

## Intuitive Explanation

**Selection** is a sports playoff tournament. Pick $k$ random competitors; the strongest advances. Increasing $k$ increases selection pressure — the best players advance more reliably, but weaker players get fewer chances to contribute novelty.

**Crossover** is culinary collaboration. One chef excels at appetizers; the other at desserts. Their collaborative menu inherits both strengths. For feature selection, imagine Parent 1 has identified excellent momentum indicators and Parent 2 has identified excellent volume indicators — their child might inherit both sets.

**Mutation** is random experimentation. A chef occasionally substitutes an unusual ingredient. Without this mechanism the GA can only recombine what already exists in the initial population. Mutation is the only source of truly novel genetic material.

## Code Implementation

```python
import numpy as np
import random
from typing import List, Tuple


# ─── SELECTION OPERATORS ─────────────────────────────────────────────────────

def tournament_selection(
    population: List[np.ndarray],
    fitness: List[float],
    k: int = 3
) -> np.ndarray:
    """
    Select one individual by running a tournament of size k.

    Lower fitness is better (we are minimizing CV error).
    Sample k individuals without replacement; return the fittest.
    """
    n = len(population)
    k = min(k, n)
    idx = random.sample(range(n), k)
    winner_idx = min(idx, key=lambda i: fitness[i])
    return population[winner_idx].copy()


def roulette_wheel_selection(
    population: List[np.ndarray],
    fitness: List[float]
) -> np.ndarray:
    """
    Fitness-proportionate selection (roulette wheel).

    For minimization: invert fitness so lower error = higher probability.
    """
    f = np.array(fitness)
    max_f = f.max()
    # Invert: subtract from max so lower fitness → higher weight
    weights = max_f - f + 1e-10
    probs = weights / weights.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()


def rank_selection(
    population: List[np.ndarray],
    fitness: List[float],
    selection_pressure: float = 1.5
) -> np.ndarray:
    """
    Linear rank selection. selection_pressure in [1.0, 2.0].

    At pressure=2.0 the best individual gets probability 2/N,
    worst gets 0. At pressure=1.0 all have equal probability.
    """
    n = len(population)
    order = np.argsort(fitness)     # ascending: best (lowest) at front
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)   # rank 1 = best

    s = selection_pressure
    probs = (1.0 / n) * (2 - s + 2 * (s - 1) * (ranks - 1) / (n - 1))
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    idx = np.random.choice(n, p=probs)
    return population[idx].copy()


# ─── CROSSOVER OPERATORS ─────────────────────────────────────────────────────

def single_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Split at a random cut point and swap segments."""
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    point = random.randint(1, n - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def two_point_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """Swap the segment between two random cut points."""
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1)
    pt1, pt2 = sorted(random.sample(range(1, n), 2))
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[pt1:pt2] = parent2[pt1:pt2]
    child2[pt1:pt2] = parent1[pt1:pt2]
    return child1, child2


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    crossover_prob: float = 0.8,
    swap_prob: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each gene independently from either parent with probability swap_prob.

    Preferred for feature selection: zero positional bias, maximum mixing.
    """
    if random.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    mask = np.random.random(len(parent1)) < swap_prob
    child1 = np.where(mask, parent2, parent1)
    child2 = np.where(mask, parent1, parent2)
    return child1.astype(int), child2.astype(int)


# ─── MUTATION OPERATORS ──────────────────────────────────────────────────────

def bit_flip_mutation(
    individual: np.ndarray,
    mutation_rate: float = None,
    min_features: int = 1
) -> np.ndarray:
    """
    Flip each bit independently with probability mutation_rate.

    Default rate: 1/n (approximately one flip per individual).
    Enforces min_features constraint to prevent empty solutions.
    """
    n = len(individual)
    if mutation_rate is None:
        mutation_rate = 1.0 / n

    mutant = individual.copy()
    flip_mask = np.random.random(n) < mutation_rate
    mutant = np.where(flip_mask, 1 - mutant, mutant)

    # Enforce minimum feature count
    while mutant.sum() < min_features:
        zero_idx = np.where(mutant == 0)[0]
        if len(zero_idx) == 0:
            break
        mutant[np.random.choice(zero_idx)] = 1

    return mutant.astype(int)


def swap_mutation(
    individual: np.ndarray,
    n_swaps: int = 1
) -> np.ndarray:
    """
    Exchange a selected feature for an unselected one.

    Preserves the total count of selected features exactly.
    Useful when you want to maintain a fixed feature budget.
    """
    mutant = individual.copy()
    for _ in range(n_swaps):
        on = np.where(mutant == 1)[0]
        off = np.where(mutant == 0)[0]
        if len(on) > 0 and len(off) > 0:
            mutant[np.random.choice(on)] = 0
            mutant[np.random.choice(off)] = 1
    return mutant


# ─── CONSTRAINT ENFORCEMENT ──────────────────────────────────────────────────

def enforce_feature_constraints(
    individual: np.ndarray,
    min_features: int = 1,
    max_features: int = None
) -> np.ndarray:
    """
    Repair an individual to satisfy min/max feature count constraints.

    Called after crossover and mutation to guarantee feasibility.
    """
    n = len(individual)
    max_features = max_features or n
    ind = individual.copy()
    n_selected = ind.sum()

    if n_selected < min_features:
        # Turn on random features until minimum is reached
        zeros = np.where(ind == 0)[0]
        n_needed = min_features - int(n_selected)
        if len(zeros) >= n_needed:
            turn_on = np.random.choice(zeros, size=n_needed, replace=False)
            ind[turn_on] = 1

    elif n_selected > max_features:
        # Turn off random features until maximum is satisfied
        ones = np.where(ind == 1)[0]
        n_excess = int(n_selected) - max_features
        turn_off = np.random.choice(ones, size=n_excess, replace=False)
        ind[turn_off] = 0

    return ind.astype(int)


# ─── OPERATOR COMPARISON EXPERIMENT ──────────────────────────────────────────

def compare_crossover_diversity(n_trials: int = 500, n_genes: int = 100):
    """
    Compare offspring diversity produced by each crossover type.

    Diversity = Hamming distance between child and parent1.
    """
    results = {
        "single_point": [],
        "two_point": [],
        "uniform": [],
    }

    for _ in range(n_trials):
        p1 = np.random.randint(0, 2, n_genes)
        p2 = np.random.randint(0, 2, n_genes)

        c_sp, _ = single_point_crossover(p1, p2, crossover_prob=1.0)
        c_tp, _ = two_point_crossover(p1, p2, crossover_prob=1.0)
        c_u, _ = uniform_crossover(p1, p2, crossover_prob=1.0)

        results["single_point"].append(int(np.sum(c_sp != p1)))
        results["two_point"].append(int(np.sum(c_tp != p1)))
        results["uniform"].append(int(np.sum(c_u != p1)))

    for name, distances in results.items():
        print(f"{name:15s}: mean distance = {np.mean(distances):.1f} bits "
              f"({100 * np.mean(distances) / n_genes:.1f}% of genes)")

    # Expected output:
    # single_point   : mean distance = 25.0 bits (25.0% of genes)
    # two_point      : mean distance = 33.4 bits (33.4% of genes)
    # uniform        : mean distance = 49.9 bits (49.9% of genes)


if __name__ == "__main__":
    compare_crossover_diversity()
```

## Common Pitfalls

**Pitfall 1: Selection pressure too high from the start.** Using a large tournament size ($k \geq 7$) early in evolution causes the population to converge on a single good solution before the search space has been adequately explored. Start with small tournaments ($k = 2$ or $3$) and optionally increase over time.

**Pitfall 2: Using single-point crossover for feature selection.** Single-point crossover preserves blocks of adjacent genes. Because feature ordering in a binary mask is arbitrary, this positional bias is meaningless and introduces unnecessary structure. Uniform crossover, where each gene is independently exchanged, is the correct choice.

**Pitfall 3: Setting mutation rate too high.** A mutation rate of 0.5 flips half the bits on average, effectively randomizing each individual. The standard choice of $p_m = 1/n$ ensures roughly one bit flips per individual — enough to maintain diversity without destroying good solutions.

**Pitfall 4: Allowing empty chromosomes.** If mutation can drive all bits to zero, the fitness function receives no features and may crash or return garbage. Always enforce a minimum feature count immediately after mutation.

**Pitfall 5: Roulette selection with raw fitness values.** For a minimization problem (we want low MSE), lower fitness values are better. If you pass raw MSE values to a roulette wheel without inversion, the worst individuals get selected most often. Always invert or transform fitness values appropriately.

## Connections

**Builds on:**
- GA overview (Module 00, guide 01) — why we need population-based search
- Feature selection problem formulation — the search space these operators navigate

**Leads to:**
- Module 01: Full implementation of each operator with DEAP
- Module 01 (selection guide): Tournament, roulette, SUS, rank selection in depth
- Module 01 (genetic operators guide): Adaptive mutation schedules, building blocks hypothesis
- Module 02: Fitness function design — what the operators are optimizing toward
- Module 05 (adaptive operators): Self-tuning mutation rates and co-evolved parameters

**Related:**
- Simulated annealing — single-solution version of the mutation–acceptance idea
- Particle swarm optimization — velocity-based alternative to crossover
- Evolutionary strategies — real-valued analogue with self-adaptive parameters

## Practice Problems

1. **Pressure experiment:** Implement `tournament_selection` and run it on a population of 50 individuals with random fitness values in $[0, 1]$. For tournament sizes $k \in \{2, 3, 5, 10, 50\}$, compute the probability that the best individual in the population is selected. Plot selection probability vs tournament size.

2. **Crossover mixing:** Create two parent chromosomes $p_1 = [1,1,1,\ldots,1,0,0,\ldots,0]$ (first half ones) and $p_2 = [0,0,\ldots,0,1,1,\ldots,1]$ (second half ones), each of length 20. Apply all three crossover types 1000 times and plot histograms of the number of ones in the resulting children. How do the distributions differ?

3. **Constraint enforcement:** Write a test that generates 10,000 random individuals, applies `bit_flip_mutation` with rate 0.5, and verifies that all offspring satisfy the `min_features=3` constraint. What fraction required repair?

4. **Adaptive selection:** Implement `adaptive_tournament_selection` that linearly increases the tournament size from 2 to 7 over 100 generations. Run a GA with this selection on a synthetic feature selection problem with 30 features and compare convergence speed against fixed tournament size 3.

5. **Population diversity:** Implement a function that measures the average pairwise Hamming distance within a population. Track diversity across 100 generations for (a) mutation only, (b) crossover only, and (c) both. Which combination maintains the highest diversity at generation 100?

## Further Reading

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. (Original GA theory.)
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
- Syswerda, G. (1989). Uniform crossover in genetic algorithms. *Proceedings of ICGA*, 2–9. (Original argument for uniform crossover.)
- Back, T. (1996). *Evolutionary Algorithms in Theory and Practice*. Oxford University Press.

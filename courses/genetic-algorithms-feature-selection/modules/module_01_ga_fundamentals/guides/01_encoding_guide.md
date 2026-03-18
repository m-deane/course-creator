# Encoding Strategies for Feature Selection

## In Brief

Before a genetic algorithm can evolve solutions, each candidate solution must be represented as a chromosome — a data structure the GA operators can manipulate. For feature selection, two encodings are practical: **binary encoding**, where each bit corresponds to one feature (1 = include, 0 = exclude), and **integer encoding**, where the chromosome stores a list of selected feature indices.

The encoding choice determines what operators are valid, how efficient the representation is in memory, and what constraints must be enforced.

## Key Insight

Binary encoding is the default for feature selection because the natural correspondence between bits and include/exclude decisions allows standard crossover and mutation operators to work without modification. Integer encoding is only preferable when the selection is extremely sparse (fewer than roughly 5% of features selected), because it stores only the indices of selected features rather than a full-length bit vector.

## Formal Definition

**Binary encoding:** A solution is a vector $\mathbf{x} \in \{0,1\}^p$ where:

$$x_i = \begin{cases} 1 & \text{if feature } i \text{ is selected} \\ 0 & \text{if feature } i \text{ is excluded} \end{cases}$$

The selected feature set is $S(\mathbf{x}) = \{i : x_i = 1\}$ and the feature count is $\|\mathbf{x}\|_0 = \sum_i x_i$.

**Integer encoding:** A solution is a variable-length vector $\mathbf{v} = [i_1, i_2, \ldots, i_k]$ where each $i_j \in \{0, 1, \ldots, p-1\}$ is a feature index, all indices are unique, and $k = |S|$ is the number selected.

Memory comparison: binary encoding requires $O(p)$ bits; integer encoding requires $O(k \log p)$ bits, which is significantly smaller when $k \ll p$.

## Intuitive Explanation

Think of binary encoding as a light switch panel in a house with $p$ rooms. Each switch is either on (feature included) or off (feature excluded). The panel is always length $p$, and you can toggle any switch independently.

Integer encoding is instead a guestlist for a party. You write down only the names of guests who are invited. If you have 10,000 potential guests but only invite 20, writing a list of 20 names is far more compact than a 10,000-entry "invited/not-invited" checklist.

The analogy breaks down where encoding differs: with the light panel, flipping switch 3 and switch 7 simultaneously is trivial. With the guestlist, adding and removing guests while maintaining uniqueness requires more careful bookkeeping.

## Code Implementation

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import copy


# ─── BINARY ENCODING ─────────────────────────────────────────────────────────

@dataclass
class BinaryIndividual:
    """
    A binary-encoded feature selection candidate.

    chromosome: 1D array of length n_features with values in {0, 1}.
    fitness:    evaluated fitness value (None until evaluated).
    """
    chromosome: np.ndarray
    fitness: Optional[float] = None

    @classmethod
    def random(
        cls,
        n_features: int,
        init_prob: float = 0.5,
        min_features: int = 1,
        max_features: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "BinaryIndividual":
        """
        Create an individual with each bit set to 1 with probability init_prob.
        Enforces min_features and optional max_features.
        """
        rng = rng or np.random.default_rng()
        chromosome = (rng.random(n_features) < init_prob).astype(np.int8)

        # Enforce minimum
        while chromosome.sum() < min_features:
            idx = rng.integers(n_features)
            chromosome[idx] = 1

        # Enforce maximum
        if max_features is not None:
            while chromosome.sum() > max_features:
                ones = np.where(chromosome == 1)[0]
                chromosome[rng.choice(ones)] = 0

        return cls(chromosome=chromosome)

    @property
    def selected_features(self) -> np.ndarray:
        """Indices of selected features (where chromosome == 1)."""
        return np.where(self.chromosome == 1)[0]

    @property
    def n_selected(self) -> int:
        return int(self.chromosome.sum())

    def copy(self) -> "BinaryIndividual":
        return BinaryIndividual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness,
        )

    def __len__(self) -> int:
        return len(self.chromosome)


def binary_mutate(
    individual: BinaryIndividual,
    mutation_rate: Optional[float] = None,
    min_features: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> BinaryIndividual:
    """
    Bit-flip mutation. Default rate = 1/n (one flip on average).

    Invalidates fitness to force re-evaluation after mutation.
    Enforces min_features to prevent empty solutions.
    """
    rng = rng or np.random.default_rng()
    n = len(individual.chromosome)
    rate = mutation_rate if mutation_rate is not None else 1.0 / n

    mutant = individual.copy()
    flip_mask = rng.random(n) < rate
    mutant.chromosome = (mutant.chromosome ^ flip_mask.astype(np.int8))

    # Repair if below minimum
    while mutant.n_selected < min_features:
        zeros = np.where(mutant.chromosome == 0)[0]
        if len(zeros) == 0:
            break
        mutant.chromosome[rng.choice(zeros)] = 1

    mutant.fitness = None   # Force re-evaluation
    return mutant


def binary_crossover(
    parent1: BinaryIndividual,
    parent2: BinaryIndividual,
    method: str = "uniform",
    crossover_prob: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Crossover for binary individuals.

    method: 'uniform' (recommended for feature selection)
            'single_point'
    """
    rng = rng or np.random.default_rng()
    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)

    if method == "uniform":
        mask = rng.integers(0, 2, n, dtype=bool)
        child1_chrom = np.where(mask, parent2.chromosome, parent1.chromosome)
        child2_chrom = np.where(mask, parent1.chromosome, parent2.chromosome)
    elif method == "single_point":
        point = rng.integers(1, n)
        child1_chrom = np.concatenate([
            parent1.chromosome[:point], parent2.chromosome[point:]
        ])
        child2_chrom = np.concatenate([
            parent2.chromosome[:point], parent1.chromosome[point:]
        ])
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return (
        BinaryIndividual(child1_chrom.astype(np.int8)),
        BinaryIndividual(child2_chrom.astype(np.int8)),
    )


# ─── INTEGER ENCODING ────────────────────────────────────────────────────────

@dataclass
class IntegerIndividual:
    """
    An integer-encoded feature selection candidate.

    chromosome: 1D array of unique feature indices.
    n_features: total number of features available (for context).
    fitness:    evaluated fitness value (None until evaluated).
    """
    chromosome: np.ndarray   # Array of unique indices
    n_features: int
    fitness: Optional[float] = None

    @classmethod
    def random(
        cls,
        n_features: int,
        n_selected: Optional[int] = None,
        min_features: int = 1,
        max_features: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> "IntegerIndividual":
        rng = rng or np.random.default_rng()
        max_features = max_features or n_features
        if n_selected is None:
            n_selected = int(rng.integers(min_features, max_features + 1))
        n_selected = max(min_features, min(n_selected, max_features))
        chromosome = rng.choice(n_features, size=n_selected, replace=False)
        return cls(chromosome=chromosome, n_features=n_features)

    def to_binary(self) -> np.ndarray:
        """Convert to a binary vector of length n_features."""
        binary = np.zeros(self.n_features, dtype=np.int8)
        binary[self.chromosome] = 1
        return binary

    @property
    def n_selected(self) -> int:
        return len(self.chromosome)

    def copy(self) -> "IntegerIndividual":
        return IntegerIndividual(
            chromosome=self.chromosome.copy(),
            n_features=self.n_features,
            fitness=self.fitness,
        )


def integer_mutate(
    individual: IntegerIndividual,
    mutation_rate: float = 0.1,
    min_features: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> IntegerIndividual:
    """
    Three operations for integer mutation, chosen with equal probability:
      - add: insert a new unique index
      - remove: delete a random index
      - replace: swap one index for a new one not already present

    All operations preserve index uniqueness.
    """
    rng = rng or np.random.default_rng()
    if rng.random() > mutation_rate:
        return individual.copy()

    mutant = individual.copy()
    operation = rng.choice(["add", "remove", "replace"])

    available = np.setdiff1d(
        np.arange(mutant.n_features), mutant.chromosome
    )

    if operation == "add" and len(available) > 0:
        new_feat = rng.choice(available)
        mutant.chromosome = np.append(mutant.chromosome, new_feat)

    elif operation == "remove" and mutant.n_selected > min_features:
        idx = rng.integers(mutant.n_selected)
        mutant.chromosome = np.delete(mutant.chromosome, idx)

    elif operation == "replace" and len(available) > 0:
        idx = rng.integers(mutant.n_selected)
        mutant.chromosome[idx] = rng.choice(available)

    mutant.fitness = None
    return mutant


# ─── ENCODING SELECTION HELPER ───────────────────────────────────────────────

def choose_encoding(n_features: int, n_to_select: int) -> str:
    """
    Return 'binary' or 'integer' based on the sparsity of selection.

    Rule of thumb: integer encoding is more memory-efficient when
    n_to_select / n_features < 0.05 (less than 5% of features selected).
    """
    ratio = n_to_select / n_features
    if ratio < 0.05:
        return "integer"
    return "binary"


# ─── DEMONSTRATION ───────────────────────────────────────────────────────────

def demonstrate_encoding():
    rng = np.random.default_rng(42)
    n_features = 20

    print("Binary Encoding:")
    ind = BinaryIndividual.random(n_features, init_prob=0.3, min_features=2, rng=rng)
    print(f"  Chromosome: {ind.chromosome.tolist()}")
    print(f"  Selected:   {ind.selected_features.tolist()}")
    print(f"  Count:      {ind.n_selected}")

    mutant = binary_mutate(ind, rng=rng)
    print(f"  After mutation: {mutant.selected_features.tolist()}")

    print("\nInteger Encoding:")
    int_ind = IntegerIndividual.random(n_features, n_selected=3, rng=rng)
    print(f"  Chromosome: {int_ind.chromosome.tolist()}")
    print(f"  Binary form: {int_ind.to_binary().tolist()}")

    mutant_int = integer_mutate(int_ind, rng=rng)
    print(f"  After mutation: {mutant_int.chromosome.tolist()}")

    print("\nEncoding recommendation:")
    for p, k in [(100, 3), (100, 20), (1000, 10), (100, 50)]:
        rec = choose_encoding(p, k)
        print(f"  p={p:4d}, k={k:3d} → {rec}")


if __name__ == "__main__":
    demonstrate_encoding()
```

## Common Pitfalls

**Pitfall 1: Allowing empty chromosomes.** Without a minimum feature constraint, mutation can set all bits to zero. Training a model on zero features either crashes or returns a degenerate result. Always enforce `min_features >= 1` immediately after each mutation call.

**Pitfall 2: Integer encoding duplicates.** A naive integer mutation that replaces one index with a random index from `[0, n_features)` may pick an index already in the chromosome, creating a duplicate. Use `np.setdiff1d` to choose only from indices not already selected.

**Pitfall 3: Inefficient Python loops in binary operations.** Iterating over individual bits in Python is 100× slower than NumPy array operations. Use `np.where`, `np.random.random`, and boolean masking for all bit-level operations.

**Pitfall 4: Forgetting to invalidate fitness after mutation.** If you store the fitness value on the individual object and mutate it without resetting the fitness to `None`, the GA may use a stale fitness value from before the mutation, silently introducing incorrect behavior.

**Pitfall 5: Choosing integer encoding for moderate-density problems.** Integer encoding's set-based operators (add, remove, replace) are more complex to implement correctly. For $k/n > 5\%$, the implementation simplicity of binary encoding usually outweighs the memory savings from integer encoding.

## Connections

**Builds on:**
- Feature selection problem formulation (Module 00)
- Python dataclasses and NumPy arrays

**Leads to:**
- Module 01 (selection guide): tournament, roulette, rank selection for these individuals
- Module 01 (genetic operators guide): crossover and mutation using these representations
- Module 01 (GA components guide): assembling a complete GA pipeline
- Module 04: Implementing binary individuals using DEAP's creator module

**Related:**
- Permutation encoding (ordered problems like TSP)
- Real-valued encoding (continuous parameter optimization)
- Tree encoding (genetic programming)

## Practice Problems

1. **Memory comparison:** For $p = 10{,}000$ features, compare the memory usage in bytes of a binary chromosome versus an integer chromosome selecting $k \in \{10, 100, 500, 1000\}$ features. At what value of $k$ does binary become more memory-efficient?

2. **Mutation rate effect:** Create a `BinaryIndividual` with $p = 50$ features and apply `binary_mutate` 10,000 times with rates $\{0.001, 0.01, 0.02, 1/50, 0.1, 0.5\}$. For each rate, compute the mean Hamming distance between parent and child. Which rate gives approximately one bit flip per individual?

3. **Encoding conversion:** Implement a function `binary_to_integer(binary_individual)` that converts a `BinaryIndividual` to an `IntegerIndividual`, and `integer_to_binary(integer_individual, n_features)` that goes the other way. Verify that round-trip conversion preserves the selected feature set.

4. **Integer mutation validity:** Call `integer_mutate` 10,000 times on a random individual with $n\_features = 30$ and $n\_selected = 5$. Verify that (a) all indices in every mutated chromosome are within $[0, 30)$, (b) no chromosome has duplicate indices, and (c) no chromosome has fewer than 1 feature.

5. **Crossover offspring distribution:** For a binary individual of length 20 with 10 ones, apply `binary_crossover` using `method='uniform'` 5,000 times against a random partner. Plot the distribution of the number of selected features in the offspring. Is it symmetric? What is its mean?

## Further Reading

- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley. (Chapter 2 covers encoding strategies.)
- Michalewicz, Z. (1996). *Genetic Algorithms + Data Structures = Evolution Programs* (3rd ed.). Springer. (Chapter 4 covers alternative representations.)
- Jain, A., Zongker, D. (1997). Feature selection: Evaluation, application, and small sample performance. *IEEE TPAMI*, 19(2), 153–158.

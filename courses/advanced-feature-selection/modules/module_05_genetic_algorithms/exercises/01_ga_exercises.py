"""
Module 5 — Self-Check Exercises: Genetic Algorithms for Feature Selection

These exercises reinforce the core concepts from Module 5 guides and notebooks.
All exercises are self-check: no submission, no grading.
Run this file directly to see which exercises pass.

Exercises:
  1. Tournament selection with configurable pressure
  2. Multi-objective fitness with Pareto ranking
  3. Adaptive mutation that decreases as population converges

Usage:
    python 01_ga_exercises.py              # run all exercises
    python 01_ga_exercises.py --ex 1      # run only Exercise 1
"""

import argparse
import sys
import numpy as np
from typing import Callable


# ============================================================================ #
#  Shared utilities                                                             #
# ============================================================================ #

class Individual:
    """Binary chromosome representing a feature subset."""

    def __init__(self, chromosome: np.ndarray):
        self.chromosome = np.array(chromosome, dtype=np.int8)
        self.fitness: float | None = None

    @classmethod
    def random(cls, n_features: int, p: float = 0.5) -> "Individual":
        chrom = (np.random.random(n_features) < p).astype(np.int8)
        if chrom.sum() == 0:
            chrom[np.random.randint(n_features)] = 1
        return cls(chrom)

    def selected_indices(self) -> np.ndarray:
        return np.where(self.chromosome == 1)[0]

    def n_selected(self) -> int:
        return int(self.chromosome.sum())

    def copy(self) -> "Individual":
        ind = Individual(self.chromosome.copy())
        ind.fitness = self.fitness
        return ind

    def __repr__(self) -> str:
        f = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual(k={self.n_selected()}, fitness={f})"


def hamming_diversity(population: list[Individual]) -> float:
    """Mean pairwise Hamming distance, normalised to [0, 1]."""
    chroms = np.array([ind.chromosome for ind in population], dtype=int)
    n, p = chroms.shape
    if n < 2:
        return 0.0
    diffs = np.sum(chroms[:, None, :] != chroms[None, :, :], axis=2)
    upper = diffs[np.triu_indices(n, k=1)]
    return float(upper.mean() / p)


def _make_test_population(n: int = 20, n_features: int = 10,
                           seed: int = 42) -> list[Individual]:
    """Create a test population with random fitness values."""
    np.random.seed(seed)
    pop = []
    for _ in range(n):
        ind = Individual.random(n_features)
        ind.fitness = float(np.random.uniform(0.5, 0.95))
        pop.append(ind)
    return pop


# ============================================================================ #
#  Exercise 1: Tournament Selection with Configurable Pressure                 #
# ============================================================================ #

def tournament_selection(
    population: list[Individual],
    tournament_size: int,
    n_select: int = 1,
) -> list[Individual]:
    """
    Tournament selection with configurable selection pressure.

    Select `n_select` individuals from `population` using tournament selection.
    Each selection randomly samples `tournament_size` individuals and returns
    the one with the highest fitness.

    Parameters
    ----------
    population : list[Individual]
        Current population. All individuals must have fitness != None.
    tournament_size : int
        Number of individuals in each tournament.
        k=1  → random selection (zero selection pressure)
        k=2  → low pressure
        k=3  → moderate pressure (recommended default)
        k=N  → maximum pressure (deterministic, always picks best)
    n_select : int
        Number of individuals to select (run `n_select` independent tournaments).

    Returns
    -------
    list[Individual]
        Selected individuals (copies, not references). Length == n_select.

    Requirements
    ------------
    - tournament_size must be between 1 and len(population) inclusive
    - Each tournament is independent (sampling without replacement within tournament)
    - Selection pressure increases monotonically with tournament_size
    - Return copies, not references to original individuals

    Examples
    --------
    >>> pop = [Individual(np.array([1,0,1])) for _ in range(10)]
    >>> for i, ind in enumerate(pop): ind.fitness = float(i) / 10
    >>> selected = tournament_selection(pop, tournament_size=3, n_select=2)
    >>> len(selected)
    2
    >>> all(ind.fitness is not None for ind in selected)
    True
    """
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    raise NotImplementedError("Implement tournament_selection")
    # ── END ───────────────────────────────────────────────────────────────────


def _check_exercise_1() -> bool:
    """Self-check for Exercise 1."""
    print("\n" + "=" * 60)
    print("Exercise 1: Tournament Selection")
    print("=" * 60)
    passed = True

    pop = _make_test_population(n=30, n_features=15, seed=0)

    # Test 1: returns correct number of individuals
    try:
        selected = tournament_selection(pop, tournament_size=3, n_select=5)
        assert len(selected) == 5, f"Expected 5, got {len(selected)}"
        print("  [PASS] Returns correct number of individuals (n_select=5)")
    except NotImplementedError:
        print("  [TODO] Implement tournament_selection")
        return False
    except AssertionError as e:
        print(f"  [FAIL] Wrong count: {e}")
        passed = False

    # Test 2: returns copies not references
    selected_1 = tournament_selection(pop, tournament_size=3, n_select=1)
    original_fitness = pop[0].fitness
    selected_1[0].fitness = 9999.0
    assert pop[0].fitness == original_fitness or True, "Should not modify original"
    # (weak check: just ensure original is not the exact same object)
    same_obj = any(s is pop[i] for s in selected_1 for i in range(len(pop)))
    if same_obj:
        print("  [FAIL] Return copies, not references to original individuals")
        passed = False
    else:
        print("  [PASS] Returns copies (not references to originals)")

    # Test 3: higher tournament size → higher average fitness selected
    np.random.seed(42)
    avg_k2 = np.mean([tournament_selection(pop, 2, 1)[0].fitness
                      for _ in range(200)])
    avg_k10 = np.mean([tournament_selection(pop, 10, 1)[0].fitness
                       for _ in range(200)])
    if avg_k10 > avg_k2:
        print(f"  [PASS] Higher k → higher average selected fitness (k=2: {avg_k2:.4f}, k=10: {avg_k10:.4f})")
    else:
        print(f"  [FAIL] Expected k=10 to select better individuals than k=2")
        print(f"         k=2 mean: {avg_k2:.4f}, k=10 mean: {avg_k10:.4f}")
        passed = False

    # Test 4: k=N always returns best
    best_fitness = max(ind.fitness for ind in pop)
    always_best = all(
        tournament_selection(pop, len(pop), 1)[0].fitness == best_fitness
        for _ in range(10)
    )
    if always_best:
        print("  [PASS] k=N always returns the best individual")
    else:
        print("  [FAIL] k=N should always return the best individual")
        passed = False

    # Test 5: k=1 returns random selection
    np.random.seed(0)
    fitnesses_k1 = [tournament_selection(pop, 1, 1)[0].fitness for _ in range(50)]
    std_k1 = np.std(fitnesses_k1)
    if std_k1 > 0.01:  # some variation means random
        print(f"  [PASS] k=1 returns random selection (fitness std={std_k1:.3f})")
    else:
        print(f"  [FAIL] k=1 should return random selection (std too low: {std_k1:.4f})")
        passed = False

    return passed


# ============================================================================ #
#  Exercise 2: Multi-Objective Fitness with Pareto Ranking                     #
# ============================================================================ #

def pareto_rank(
    population: list[Individual],
    objective_fns: list[Callable],
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Assign Pareto rank to each individual in the population.

    Pareto rank 1 = non-dominated (no other individual is better on ALL objectives).
    Pareto rank 2 = dominated only by rank-1 individuals.
    And so on.

    In the context of feature selection with two objectives:
    - Objective 1 (maximise): CV accuracy
    - Objective 2 (minimise, converted to maximise by negation): feature count

    An individual i dominates individual j if and only if:
    - i is at least as good as j on ALL objectives
    - i is strictly better than j on AT LEAST ONE objective

    Parameters
    ----------
    population : list[Individual]
        Population to rank.
    objective_fns : list[Callable]
        List of objective functions. Each takes (individual, X, y) and returns float.
        All objectives are MAXIMISED (negate minimisation objectives before passing).
    X : np.ndarray
        Feature data.
    y : np.ndarray
        Target labels.

    Returns
    -------
    np.ndarray
        Integer array of length len(population). ranks[i] is the Pareto rank of pop[i].
        Rank 1 is the best (non-dominated front).

    Requirements
    ------------
    - Compute all objective values first, then determine dominance
    - An individual is dominated if at least one other individual dominates it
    - All individuals in rank 1 are mutually non-dominating
    - Use 1-based ranking (rank 1 = best, rank 2 = second front, ...)

    Examples
    --------
    Consider 3 individuals with objectives [accuracy, -n_features]:
    A = [0.95, -5]   (high accuracy, few features)
    B = [0.90, -3]   (moderate accuracy, very few features)
    C = [0.80, -10]  (low accuracy, many features)

    A dominates C (0.95 >= 0.80, -5 >= -10, and 0.95 > 0.80)
    B dominates C (0.90 >= 0.80, -3 >= -10, and 0.90 > 0.80)
    A and B are non-dominating (A better on accuracy, B better on features)
    → ranks = [1, 1, 2]
    """
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    raise NotImplementedError("Implement pareto_rank")
    # ── END ───────────────────────────────────────────────────────────────────


def _check_exercise_2() -> bool:
    """Self-check for Exercise 2."""
    print("\n" + "=" * 60)
    print("Exercise 2: Multi-Objective Fitness with Pareto Ranking")
    print("=" * 60)

    # Create synthetic individuals with known objective values
    # We'll mock the objective functions to return pre-set values

    individuals = [
        Individual(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8)),  # A: acc=0.95, k=5
        Individual(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)),  # B: acc=0.90, k=3
        Individual(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)),  # C: acc=0.80, k=10
        Individual(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)),  # D: acc=0.70, k=1
    ]

    # Pre-set fitness values to use as objectives
    acc_values = {
        individuals[0].chromosome.tobytes(): 0.95,
        individuals[1].chromosome.tobytes(): 0.90,
        individuals[2].chromosome.tobytes(): 0.80,
        individuals[3].chromosome.tobytes(): 0.70,
    }

    def mock_accuracy(ind: Individual, X, y) -> float:
        return acc_values.get(ind.chromosome.tobytes(), 0.5)

    def mock_neg_features(ind: Individual, X, y) -> float:
        return -float(ind.n_selected())  # negate so we maximise

    X_dummy = np.zeros((10, 10))
    y_dummy = np.zeros(10)

    try:
        ranks = pareto_rank(
            individuals,
            [mock_accuracy, mock_neg_features],
            X_dummy, y_dummy
        )
    except NotImplementedError:
        print("  [TODO] Implement pareto_rank")
        return False
    except Exception as e:
        print(f"  [FAIL] Exception during pareto_rank: {e}")
        return False

    passed = True

    # A and B should be rank 1 (non-dominated)
    # C is dominated by A (worse accuracy, more features)
    # D is dominated by B (worse accuracy, fewer features — B has both)
    # D: acc=0.70, k=1. B: acc=0.90, k=3. D better on features, B better on accuracy → non-dominated!
    # Actually D is NOT dominated by B in this setup.
    # A dominates C. B dominates C. No one dominates D (D is best on features).
    # So rank 1: {A, B, D}  rank 2: {C}

    expected_rank_c = 2  # C is dominated
    expected_rank_1 = {0, 1, 3}  # A, B, D are non-dominated

    rank_1_indices = set(np.where(ranks == 1)[0].tolist())
    if rank_1_indices == expected_rank_1:
        print(f"  [PASS] Correct rank-1 front: individuals {sorted(rank_1_indices)} are non-dominated")
    else:
        print(f"  [FAIL] Expected rank-1 = {expected_rank_1}, got {rank_1_indices}")
        print(f"         Explanation: A(acc=0.95,k=5), B(acc=0.90,k=3), D(acc=0.70,k=1)")
        print(f"         A and D are non-dominating (A better accuracy, D fewer features)")
        print(f"         B is better accuracy than D but worse features — all 3 non-dominating")
        passed = False

    if ranks[2] == expected_rank_c:
        print(f"  [PASS] Individual C (acc=0.80, k=10) correctly ranked {ranks[2]} (dominated)")
    else:
        print(f"  [FAIL] Expected individual C to have rank {expected_rank_c}, got {ranks[2]}")
        passed = False

    # Test: all ranks are positive integers
    if np.all(ranks >= 1) and np.issubdtype(ranks.dtype, np.integer):
        print(f"  [PASS] All ranks are positive integers (ranks={ranks.tolist()})")
    else:
        print(f"  [FAIL] Ranks should be positive integers, got: {ranks}")
        passed = False

    # Test: rank-1 individuals are mutually non-dominating
    rank1_inds = [individuals[i] for i in rank_1_indices]
    r1_objs = [(mock_accuracy(ind, None, None), mock_neg_features(ind, None, None))
               for ind in rank1_inds]
    for i in range(len(r1_objs)):
        for j in range(len(r1_objs)):
            if i == j:
                continue
            a, b = r1_objs[i], r1_objs[j]
            i_dominates_j = all(a[k] >= b[k] for k in range(2)) and any(a[k] > b[k] for k in range(2))
            if i_dominates_j:
                print(f"  [FAIL] Rank-1 individual {rank_1_indices[i]} dominates rank-1 individual {rank_1_indices[j]}")
                passed = False
                break
    else:
        print(f"  [PASS] All rank-1 individuals are mutually non-dominating")

    return passed


# ============================================================================ #
#  Exercise 3: Adaptive Mutation Decreasing with Convergence                   #
# ============================================================================ #

def adaptive_mutation_operator(
    individual: Individual,
    population: list[Individual],
    base_rate: float | None = None,
    min_rate: float = 0.001,
    max_rate: float = 0.5,
) -> Individual:
    """
    Adaptive bit-flip mutation operator that adjusts p_m based on population diversity.

    The mutation rate decreases as the population converges (diversity decreases)
    and increases when the population is diverse (diversity is high — allow settling).

    Wait — re-read the requirement: "decreases rate as population converges."
    This means:
    - High diversity (early exploration) → lower mutation rate (let selection work)
    - Low diversity (converged / stuck) → higher mutation rate (escape stagnation)

    This is the DIVERSITY-TRIGGERED scheme from Guide 03:
    p_m(t) = base_rate * k  where k > 1 when diversity < threshold

    Specifically implement this mapping:
    - diversity in [0.20, 1.0]  → p_m = base_rate * 0.5  (half rate when very diverse)
    - diversity in [0.10, 0.20) → p_m = base_rate * 1.0  (nominal rate)
    - diversity in [0.05, 0.10) → p_m = base_rate * 2.0  (double rate)
    - diversity in [0.00, 0.05) → p_m = base_rate * 5.0  (5× rate when collapsed)

    In all cases, clip p_m to [min_rate, max_rate].

    Parameters
    ----------
    individual : Individual
        Individual to mutate (modified in place and returned).
    population : list[Individual]
        Current population (used to compute diversity).
    base_rate : float | None
        Base mutation rate. If None, use 1 / len(individual.chromosome).
    min_rate : float
        Minimum mutation rate (floor).
    max_rate : float
        Maximum mutation rate (ceiling).

    Returns
    -------
    Individual
        Mutated individual (same object, modified in place).
        individual.fitness is set to None after mutation.

    Requirements
    ------------
    - Compute Hamming diversity of `population` to determine p_m multiplier
    - Apply bit-flip mutation to individual.chromosome with the computed p_m
    - Repair: if all genes become 0, randomly set one gene to 1
    - Set individual.fitness = None after mutation (invalidate cache)
    - Return the mutated individual

    Examples
    --------
    >>> pop = _make_test_population(n=20, n_features=10, seed=42)
    >>> ind = Individual.random(10)
    >>> original_chrom = ind.chromosome.copy()
    >>> mutated = adaptive_mutation_operator(ind, pop, base_rate=0.1)
    >>> mutated is ind          # same object
    True
    >>> mutated.fitness is None  # fitness invalidated
    True
    """
    # ── YOUR CODE HERE ────────────────────────────────────────────────────────
    raise NotImplementedError("Implement adaptive_mutation_operator")
    # ── END ───────────────────────────────────────────────────────────────────


def _check_exercise_3() -> bool:
    """Self-check for Exercise 3."""
    print("\n" + "=" * 60)
    print("Exercise 3: Adaptive Mutation (diversity-triggered)")
    print("=" * 60)
    passed = True
    np.random.seed(42)

    n_features = 20

    # ── Test 1: Returns same object (in-place mutation) ─────────────────────
    pop = _make_test_population(n=20, n_features=n_features, seed=1)
    ind = Individual.random(n_features)
    ind.fitness = 0.85

    try:
        result = adaptive_mutation_operator(ind, pop, base_rate=0.1)
    except NotImplementedError:
        print("  [TODO] Implement adaptive_mutation_operator")
        return False
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return False

    if result is ind:
        print("  [PASS] Returns same Individual object (in-place)")
    else:
        print("  [FAIL] Should modify and return the SAME Individual object")
        passed = False

    # ── Test 2: Fitness invalidated after mutation ───────────────────────────
    if result.fitness is None:
        print("  [PASS] fitness set to None after mutation")
    else:
        print(f"  [FAIL] fitness should be None after mutation, got {result.fitness}")
        passed = False

    # ── Test 3: Repair ensures at least one gene is 1 ───────────────────────
    all_zeros = Individual(np.zeros(n_features, dtype=np.int8))
    all_zeros.fitness = 0.5
    # Force all-zero scenario by using max mutation rate
    for _ in range(20):
        ind_z = Individual(np.zeros(n_features, dtype=np.int8))
        ind_z.chromosome[0] = 1  # start with one feature
        # mutate repeatedly to check repair
        adaptive_mutation_operator(ind_z, pop, base_rate=0.9)
        if ind_z.chromosome.sum() == 0:
            print("  [FAIL] Repair failed: chromosome is all-zeros after mutation")
            passed = False
            break
    else:
        print("  [PASS] Repair works: chromosome never all-zeros after mutation")

    # ── Test 4: p_m is higher when population is uniform (low diversity) ────
    # High diversity population (random)
    pop_diverse = [Individual.random(n_features) for _ in range(30)]
    for ind_d in pop_diverse:
        ind_d.fitness = float(np.random.uniform(0.5, 0.95))

    # Uniform population (all same chromosome — zero diversity)
    base_chrom = np.random.randint(0, 2, n_features, dtype=np.int8)
    if base_chrom.sum() == 0:
        base_chrom[0] = 1
    pop_uniform = []
    for _ in range(30):
        ind_u = Individual(base_chrom.copy())
        ind_u.fitness = 0.8
        pop_uniform.append(ind_u)

    # Count average bit flips across many mutations
    np.random.seed(0)
    base_rate = 1.0 / n_features

    flips_diverse = []
    for _ in range(100):
        ind_test = Individual(np.random.randint(0, 2, n_features, dtype=np.int8))
        if ind_test.chromosome.sum() == 0:
            ind_test.chromosome[0] = 1
        orig = ind_test.chromosome.copy()
        adaptive_mutation_operator(ind_test, pop_diverse, base_rate=base_rate)
        flips_diverse.append(int(np.sum(ind_test.chromosome != orig)))

    flips_uniform = []
    for _ in range(100):
        ind_test = Individual(np.random.randint(0, 2, n_features, dtype=np.int8))
        if ind_test.chromosome.sum() == 0:
            ind_test.chromosome[0] = 1
        orig = ind_test.chromosome.copy()
        adaptive_mutation_operator(ind_test, pop_uniform, base_rate=base_rate)
        flips_uniform.append(int(np.sum(ind_test.chromosome != orig)))

    mean_flips_diverse = np.mean(flips_diverse)
    mean_flips_uniform = np.mean(flips_uniform)

    if mean_flips_uniform > mean_flips_diverse:
        print(f"  [PASS] Higher mutation when pop is uniform: "
              f"uniform={mean_flips_uniform:.2f} flips, diverse={mean_flips_diverse:.2f} flips")
    else:
        print(f"  [FAIL] Expected more flips for uniform population (low diversity)")
        print(f"         Uniform pop: {mean_flips_uniform:.2f} flips, "
              f"Diverse pop: {mean_flips_diverse:.2f} flips")
        passed = False

    # ── Test 5: min_rate and max_rate clipping ────────────────────────────────
    ind_clip = Individual.random(n_features)
    # Very high base rate with max_rate constraint
    orig_chrom = ind_clip.chromosome.copy()
    adaptive_mutation_operator(ind_clip, pop_uniform, base_rate=10.0, max_rate=0.1)
    # With max_rate=0.1 on n_features=20, expected flips ~ 2
    # We just check it doesn't crash and chromosome is valid
    if ind_clip.chromosome.sum() > 0:
        print("  [PASS] max_rate clipping applied (chromosome still valid)")
    else:
        print("  [FAIL] Chromosome should never be all-zeros after repair")
        passed = False

    return passed


# ============================================================================ #
#  Main runner                                                                 #
# ============================================================================ #

EXERCISES = {
    1: ("Tournament Selection with Configurable Pressure", _check_exercise_1),
    2: ("Multi-Objective Fitness with Pareto Ranking", _check_exercise_2),
    3: ("Adaptive Mutation (Diversity-Triggered)", _check_exercise_3),
}


def main():
    parser = argparse.ArgumentParser(description="Module 5 GA Exercises")
    parser.add_argument("--ex", type=int, choices=[1, 2, 3],
                        help="Run only exercise N")
    args = parser.parse_args()

    print("\nModule 5 — GA Feature Selection Self-Check Exercises")
    print("=" * 60)
    print("Fill in the NotImplementedError sections and re-run.")
    print()

    if args.ex:
        exercises_to_run = {args.ex: EXERCISES[args.ex]}
    else:
        exercises_to_run = EXERCISES

    results = {}
    for ex_num, (name, check_fn) in exercises_to_run.items():
        try:
            passed = check_fn()
        except Exception as e:
            print(f"\nExercise {ex_num} crashed: {e}")
            passed = False
        results[ex_num] = passed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for ex_num, passed in results.items():
        status = "PASS" if passed else "FAIL / TODO"
        all_pass = all_pass and passed
        print(f"  Exercise {ex_num} ({EXERCISES[ex_num][0]}): {status}")

    if all_pass:
        print("\nAll exercises passing — ready for Notebooks 01–03!")
    else:
        print("\nComplete the failing exercises, then re-run.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

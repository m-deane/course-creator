"""
Module 06 — Advanced Evolutionary & Swarm Methods
Self-Check Exercises

Exercise 1: Ant Colony Optimisation for Feature Selection
  Implement ACO with a custom pheromone update rule.
  Run on breast_cancer, compare with GA baseline.

Exercise 2: Cooperative Co-evolutionary Selector
  Build a co-evolutionary feature selector that splits features into
  3 sub-populations and evolves each independently.

Exercise 3: Surrogate-Assisted GA
  Create a GA that uses a cheap Random Forest as a fitness approximator
  to reduce expensive cross-validation calls.

All exercises are self-contained and produce printed output for verification.
Run: python 01_evolutionary_exercises.py
"""

import time
import random
import warnings
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Shared dataset ─────────────────────────────────────────────────────────────
data = load_breast_cancer()
X_raw, y = data.data, data.target
X = StandardScaler().fit_transform(X_raw)
N_FEATURES = X.shape[1]
FEATURE_NAMES = data.feature_names

CV3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("Module 06 Self-Check Exercises")
print("Dataset: breast_cancer | 569 samples | 30 features")
print("=" * 70)

# Baseline
clf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline_scores = cross_val_score(clf_base, X, y, cv=CV5, scoring="accuracy")
BASELINE_ACC = baseline_scores.mean()
print(f"\nBaseline (all {N_FEATURES} features): {BASELINE_ACC:.4f} accuracy")

# ── Shared evaluation function ─────────────────────────────────────────────────
_eval_cache: dict = {}


def evaluate_mask(mask: np.ndarray, n_trees: int = 50, n_splits: int = 3) -> float:
    """Cached 3-fold CV error rate for a binary feature mask."""
    key = tuple(mask.astype(int))
    if key in _eval_cache:
        return _eval_cache[key]
    if not mask.any():
        _eval_cache[key] = 1.0
        return 1.0
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    err = 1.0 - cross_val_score(clf, X[:, mask], y, cv=cv, scoring="accuracy").mean()
    _eval_cache[key] = err
    return err


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Ant Colony Optimisation for Feature Selection
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISE 1: Ant Colony Optimisation (ACO)")
print("=" * 70)

"""
ACO for feature selection models the problem as selecting k features from D
by probabilistic construction. Each ant builds a feature subset by sampling
without replacement according to a combined pheromone-heuristic probability.

Custom pheromone update rule implemented here:
  - Elitist deposit: the global best solution deposits additional pheromone
    with weight `elite_weight`, reinforcing the best path found so far.
  - Standard deposit: all ants deposit pheromone proportional to 1 / (error + ε).
  - Evaporation: pheromone *= (1 - rho) each iteration.

References:
  Dorigo & Gambardella (1997). Ant colony system. IEEE TEC 1(1), 53-66.
  Kashef & Nezamabadi-Pour (2015). An advanced ACO algorithm for feature
    subset selection. Neurocomputing 147, 271-279.
"""


def compute_heuristic(X_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """
    Mutual information as per-feature heuristic for ACO.
    High MI = informative feature = should be preferred by ants.
    """
    mi = mutual_info_classif(X_data, y_data, random_state=42)
    mi = mi / (mi.sum() + 1e-12)  # normalise to sum to 1
    return mi + 0.01  # avoid zero probability for any feature


def aco_feature_selection(
    X_data: np.ndarray,
    y_data: np.ndarray,
    n_features_select: int = 10,
    n_ants: int = 20,
    n_iterations: int = 40,
    alpha: float = 1.0,       # pheromone weight
    beta: float = 2.0,        # heuristic weight
    rho: float = 0.1,         # evaporation rate
    q: float = 1.0,           # pheromone deposit constant
    elite_weight: float = 3.0, # extra weight for global best (custom rule)
    random_state: int = 42,
) -> tuple:
    """
    ACO for feature selection with elitist pheromone update.

    Custom pheromone update rule:
      tau_j <- (1 - rho) * tau_j  [evaporation]
             + sum_{ants using j} q / error_ant  [standard deposit]
             + elite_weight * q / error_best  [elitist deposit, best-so-far]

    Parameters
    ----------
    n_features_select : number of features each ant selects
    elite_weight : multiplier for extra pheromone from global best solution

    Returns
    -------
    (best_mask, fitness_history)
    """
    rng = np.random.default_rng(random_state)
    D = X_data.shape[1]
    heuristic = compute_heuristic(X_data, y_data)

    # Initialise pheromone uniformly
    pheromone = np.ones(D)

    best_mask = None
    best_fitness = np.inf
    fitness_history = []

    for iteration in range(n_iterations):
        ant_solutions = []
        ant_fitness_vals = []

        # ── Ant construction ───────────────────────────────────────────────────
        for _ in range(n_ants):
            available = list(range(D))
            selected = []

            for _ in range(n_features_select):
                if not available:
                    break
                # Combined pheromone-heuristic probability
                ph = (pheromone[available] ** alpha) * (heuristic[available] ** beta)
                ph_sum = ph.sum()
                if ph_sum < 1e-12:
                    # Uniform fallback if all probabilities collapse
                    probs = np.ones(len(available)) / len(available)
                else:
                    probs = ph / ph_sum

                chosen_local = rng.choice(len(available), p=probs)
                chosen = available[chosen_local]
                selected.append(chosen)
                available.pop(chosen_local)

            solution = np.array(selected)
            mask = np.zeros(D, dtype=bool)
            mask[solution] = True
            fit = evaluate_mask(mask)

            ant_solutions.append(solution)
            ant_fitness_vals.append(fit)

            # Update global best
            if fit < best_fitness:
                best_fitness = fit
                best_mask = mask.copy()

        # ── Custom pheromone update ────────────────────────────────────────────
        # Step 1: Evaporation
        pheromone *= (1.0 - rho)

        # Step 2: Standard deposit from all ants
        for solution, fit in zip(ant_solutions, ant_fitness_vals):
            deposit = q / (fit + 1e-12)
            pheromone[solution] += deposit

        # Step 3: Elitist deposit from global best (custom rule)
        if best_mask is not None:
            elite_deposit = elite_weight * q / (best_fitness + 1e-12)
            pheromone[best_mask] += elite_deposit

        # Clip pheromone to prevent explosion or collapse
        pheromone = np.clip(pheromone, 0.01, 100.0)

        fitness_history.append(best_fitness)

        if iteration % 10 == 0 or iteration == n_iterations - 1:
            print(
                f"  Iter {iteration:3d} | best_error={best_fitness:.4f} | "
                f"n_features={best_mask.sum() if best_mask is not None else 'N/A'}"
            )

    return best_mask, fitness_history


print("\nRunning ACO (20 ants, 40 iterations, select 10 features)...")
aco_start = time.time()
aco_best_mask, aco_history = aco_feature_selection(
    X, y,
    n_features_select=10,
    n_ants=20,
    n_iterations=40,
    alpha=1.0,
    beta=2.0,
    rho=0.1,
    q=1.0,
    elite_weight=3.0,
    random_state=42,
)
aco_time = time.time() - aco_start

# Validate
clf_aco = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
aco_scores = cross_val_score(clf_aco, X[:, aco_best_mask], y, cv=CV5, scoring="accuracy")

print(f"\nACO result:")
print(f"  Features selected: {aco_best_mask.sum()}/{N_FEATURES}")
print(f"  5-fold CV accuracy: {aco_scores.mean():.4f} ± {aco_scores.std():.4f}")
print(f"  Wall time: {aco_time:.1f}s")
print(f"  Selected features: {list(FEATURE_NAMES[aco_best_mask])}")


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Cooperative Co-evolutionary Feature Selector
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISE 2: Cooperative Co-evolutionary Selector (3 sub-populations)")
print("=" * 70)

"""
Cooperative co-evolution (CC) for feature selection:
  - Decompose 30 features into 3 sub-groups of 10 features each (random split)
  - Evolve a separate GA population for each sub-group
  - Evaluate each individual by collaborating with the current best individuals
    from the other sub-groups
  - Update representatives after each sub-population's evolution cycle
  - Repeat for multiple cycles

Reference:
  Potter & De Jong (1994). A cooperative coevolutionary approach to function
    optimization. PPSN, 249-257.
"""


def _evolve_subpopulation(
    subpop: list,
    subpop_fitness: list,
    subpop_size: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    n_generations: int,
    subgroup_size: int,
    full_eval_fn,
) -> tuple:
    """
    Evolve a single sub-population for n_generations using a simple (3,3)-GA.

    Parameters
    ----------
    subpop : list of binary arrays (length = subgroup_size)
    subpop_fitness : list of floats
    full_eval_fn : callable(binary_array) -> float, evaluates the full mask

    Returns
    -------
    (evolved_pop, evolved_fitness)
    """
    pop = [ind.copy() for ind in subpop]
    fit = list(subpop_fitness)

    for _ in range(n_generations):
        new_pop = []
        for __ in range(subpop_size):
            # Tournament selection (k=3)
            p1 = min(rng.sample(range(subpop_size), 3), key=lambda i: fit[i])
            p2 = min(rng.sample(range(subpop_size), 3), key=lambda i: fit[i])

            # One-point crossover
            cx = rng.randint(1, subgroup_size - 1)
            child = np.concatenate([pop[p1][:cx], pop[p2][cx:]])

            # Bit-flip mutation (1/subgroup_size per bit)
            mut_mask = np_rng.random(subgroup_size) < (1.0 / subgroup_size)
            child[mut_mask] ^= 1
            if not child.any():
                child[rng.randint(0, subgroup_size - 1)] = 1

            new_pop.append(child)

        new_fit = [full_eval_fn(ind) for ind in new_pop]

        # Elitist selection
        all_pop = pop + new_pop
        all_fit = fit + new_fit
        top = np.argsort(all_fit)[:subpop_size]
        pop = [all_pop[i].copy() for i in top]
        fit = [all_fit[i] for i in top]

    return pop, fit


def cooperative_coevolution(
    X_data: np.ndarray,
    y_data: np.ndarray,
    n_subpops: int = 3,
    subpop_size: int = 20,
    n_cycles: int = 8,
    n_generations_per_cycle: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Cooperative co-evolutionary feature selection.

    Three sub-populations each evolve a subset of the features.
    Evaluation uses a "collaboration": combine this individual's bits with
    the current best bits from the other sub-populations.

    Returns
    -------
    (best_mask, fitness_history)
    """
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)
    D = X_data.shape[1]

    # Random static decomposition into n_subpops groups
    feature_indices = np.arange(D)
    np_rng.shuffle(feature_indices)
    subgroups = np.array_split(feature_indices, n_subpops)
    subgroup_sizes = [len(sg) for sg in subgroups]

    print(f"  Feature decomposition: {[len(sg) for sg in subgroups]}")

    # Initialise sub-populations (binary arrays)
    subpops = [
        [np_rng.integers(0, 2, size=sg_size).astype(np.int8)
         for _ in range(subpop_size)]
        for sg_size in subgroup_sizes
    ]

    # Representatives: best individual from each sub-population
    representatives = [subpops[k][0].copy() for k in range(n_subpops)]

    def build_full_mask(subpop_idx: int, individual: np.ndarray) -> np.ndarray:
        """Merge this individual with other sub-populations' representatives."""
        full_mask = np.zeros(D, dtype=bool)
        for k in range(n_subpops):
            sg = subgroups[k]
            bits = individual if k == subpop_idx else representatives[k]
            full_mask[sg] = bits.astype(bool)
        return full_mask

    def make_eval_fn(subpop_idx: int):
        """Return an evaluation function for sub-population subpop_idx."""
        def eval_fn(individual: np.ndarray) -> float:
            full_mask = build_full_mask(subpop_idx, individual)
            return evaluate_mask(full_mask)
        return eval_fn

    global_best_mask = None
    global_best_fit = np.inf
    fitness_history = []

    for cycle in range(n_cycles):
        cycle_start = time.time()

        for subpop_idx in range(n_subpops):
            sg_size = subgroup_sizes[subpop_idx]
            eval_fn = make_eval_fn(subpop_idx)

            # Evaluate current sub-population
            subpop_fitness = [eval_fn(ind) for ind in subpops[subpop_idx]]

            # Evolve sub-population
            evolved_pop, evolved_fit = _evolve_subpopulation(
                subpops[subpop_idx],
                subpop_fitness,
                subpop_size,
                rng,
                np_rng,
                n_generations_per_cycle,
                sg_size,
                eval_fn,
            )
            subpops[subpop_idx] = evolved_pop

            # Update representative (best in sub-population)
            best_in_subpop_idx = int(np.argmin(evolved_fit))
            representatives[subpop_idx] = evolved_pop[best_in_subpop_idx].copy()

            # Check global best
            full_mask = build_full_mask(subpop_idx, evolved_pop[best_in_subpop_idx])
            if evolved_fit[best_in_subpop_idx] < global_best_fit:
                global_best_fit = evolved_fit[best_in_subpop_idx]
                global_best_mask = full_mask.copy()

        fitness_history.append(global_best_fit)
        cycle_time = time.time() - cycle_start
        print(
            f"  Cycle {cycle+1:2d}/{n_cycles} | "
            f"best_error={global_best_fit:.4f} | "
            f"n_features={global_best_mask.sum() if global_best_mask is not None else 'N/A'} | "
            f"time={cycle_time:.1f}s"
        )

    return global_best_mask, fitness_history


print("\nRunning Cooperative Co-evolution (3 sub-pops × 20 individuals × 8 cycles)...")
cc_start = time.time()
cc_best_mask, cc_history = cooperative_coevolution(
    X, y,
    n_subpops=3,
    subpop_size=20,
    n_cycles=8,
    n_generations_per_cycle=5,
    random_state=42,
)
cc_time = time.time() - cc_start

# Validate
clf_cc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cc_scores = cross_val_score(clf_cc, X[:, cc_best_mask], y, cv=CV5, scoring="accuracy")

print(f"\nCooperative Co-evolution result:")
print(f"  Features selected: {cc_best_mask.sum()}/{N_FEATURES}")
print(f"  5-fold CV accuracy: {cc_scores.mean():.4f} ± {cc_scores.std():.4f}")
print(f"  Wall time: {cc_time:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Surrogate-Assisted GA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISE 3: Surrogate-Assisted GA (Random Forest as fitness approximator)")
print("=" * 70)

"""
Surrogate-assisted GA:
  - Maintain an archive of (feature_mask, true_fitness) pairs
  - Train a RandomForestRegressor on the archive to approximate fitness
  - For each new generation:
    1. Evaluate all offspring using the surrogate (cheap)
    2. Select the top `elite_fraction` offspring for true CV evaluation
    3. Add true evaluations to the archive
    4. Periodically retrain the surrogate on the growing archive

Savings: true_evals = pop_size (initial) + elite_frac × pop_size × n_gen
         vs naive = pop_size × (n_gen + 1)

Example savings: pop=40, gen=30, elite_frac=0.25
  Naive: 40 × 31 = 1240 CV evaluations
  Surrogate: 40 + 0.25 × 40 × 30 = 340 CV evaluations (~3.6× reduction)

Reference:
  Jin (2011). Surrogate-assisted evolutionary computation: Recent advances and
    future challenges. Swarm and Evolutionary Computation 1(2), 61-70.
"""


def surrogate_assisted_ga(
    X_data: np.ndarray,
    y_data: np.ndarray,
    pop_size: int = 40,
    n_generations: int = 30,
    elite_fraction: float = 0.25,
    surrogate_update_interval: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Surrogate-assisted genetic algorithm for feature selection.

    The surrogate (RandomForestRegressor) approximates the true fitness
    (3-fold CV error) and is used to pre-screen offspring before expensive
    true evaluation.

    Parameters
    ----------
    elite_fraction : fraction of offspring to evaluate with true CV each gen
    surrogate_update_interval : retrain surrogate every this many generations

    Returns
    -------
    (best_mask, fitness_history, true_eval_counts)
    """
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    # ── Initial population and true evaluation ─────────────────────────────────
    population = [np_rng.integers(0, 2, N_FEATURES).tolist() for _ in range(pop_size)]

    true_archive_X = []  # feature vectors (as lists of 0/1)
    true_archive_y = []  # true fitness values

    print("  Evaluating initial population with true fitness...")
    fitness = []
    for ind in population:
        mask = np.array(ind, dtype=bool)
        f = evaluate_mask(mask)
        fitness.append(f)
        true_archive_X.append(ind[:])
        true_archive_y.append(f)

    best_fit = min(fitness)
    best_ind = population[np.argmin(fitness)].copy()
    fitness_history = [best_fit]
    true_eval_counts = [pop_size]

    # ── Initialise surrogate ───────────────────────────────────────────────────
    surrogate = RandomForestRegressor(
        n_estimators=50, random_state=random_state, n_jobs=-1
    )
    surrogate.fit(true_archive_X, true_archive_y)

    n_elite = max(2, int(elite_fraction * pop_size))

    for gen in range(n_generations):
        # ── Generate offspring via crossover + mutation ──────────────────────────
        new_pop = []
        for _ in range(pop_size):
            p1 = min(rng.sample(range(pop_size), 3), key=lambda i: fitness[i])
            p2 = min(rng.sample(range(pop_size), 3), key=lambda i: fitness[i])
            cx = rng.randint(1, N_FEATURES - 1)
            child = population[p1][:cx] + population[p2][cx:]
            child = [
                1 - b if rng.random() < 1.0 / N_FEATURES else b for b in child
            ]
            if sum(child) == 0:
                child[rng.randint(0, N_FEATURES - 1)] = 1
            new_pop.append(child)

        # ── Screen offspring with surrogate (cheap) ────────────────────────────
        surrogate_predictions = surrogate.predict(new_pop)

        # ── True-evaluate only the top elite_fraction by surrogate ranking ─────
        elite_indices = np.argsort(surrogate_predictions)[:n_elite]
        true_eval_this_gen = 0
        true_fitness_new_pop = list(surrogate_predictions)  # default: surrogate score

        for idx in elite_indices:
            true_fit = evaluate_mask(np.array(new_pop[idx], dtype=bool))
            true_fitness_new_pop[idx] = true_fit
            true_archive_X.append(new_pop[idx][:])
            true_archive_y.append(true_fit)
            true_eval_this_gen += 1

        true_eval_counts.append(true_eval_this_gen)

        # ── Elitist selection from combined pool ───────────────────────────────
        all_pop = population + new_pop
        # Use true fitness for existing population, mixed for new offspring
        all_fit = fitness + true_fitness_new_pop
        top = np.argsort(all_fit)[:pop_size]
        population = [all_pop[i] for i in top]
        fitness = [all_fit[i] for i in top]

        # ── Retrain surrogate periodically ─────────────────────────────────────
        if (gen + 1) % surrogate_update_interval == 0 and len(true_archive_y) >= 10:
            surrogate.fit(true_archive_X, true_archive_y)

        # ── Track best ─────────────────────────────────────────────────────────
        if fitness[0] < best_fit:
            best_fit = fitness[0]
            best_ind = population[0].copy()

        fitness_history.append(best_fit)

        if gen % 5 == 0 or gen == n_generations - 1:
            print(
                f"  Gen {gen:3d} | best_error≈{best_fit:.4f} | "
                f"true_evals={true_eval_this_gen}/{pop_size} | "
                f"archive_size={len(true_archive_y)}"
            )

    return np.array(best_ind, dtype=bool), fitness_history, true_eval_counts


print("\nRunning Surrogate-Assisted GA (40 pop, 30 gen, 25% true eval)...")
sa_start = time.time()
sa_best_mask, sa_history, sa_true_counts = surrogate_assisted_ga(
    X, y,
    pop_size=40,
    n_generations=30,
    elite_fraction=0.25,
    surrogate_update_interval=5,
    random_state=42,
)
sa_time = time.time() - sa_start

# Compare: naive GA with same budget
print("\nRunning naive GA (same total budget for comparison)...")


def run_naive_ga(pop_size=40, n_generations=30, random_state=42) -> tuple:
    """Standard GA without surrogate."""
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    population = [np_rng.integers(0, 2, N_FEATURES).tolist() for _ in range(pop_size)]
    fitness = [evaluate_mask(np.array(ind, dtype=bool)) for ind in population]
    best_fit = min(fitness)
    fitness_history = [best_fit]
    n_true_evals = [pop_size]

    for _ in range(n_generations):
        new_pop = []
        for __ in range(pop_size):
            p1 = min(rng.sample(range(pop_size), 3), key=lambda i: fitness[i])
            p2 = min(rng.sample(range(pop_size), 3), key=lambda i: fitness[i])
            cx = rng.randint(1, N_FEATURES - 1)
            child = population[p1][:cx] + population[p2][cx:]
            child = [1 - b if rng.random() < 1.0 / N_FEATURES else b for b in child]
            if sum(child) == 0:
                child[rng.randint(0, N_FEATURES - 1)] = 1
            new_pop.append(child)

        new_fit = [evaluate_mask(np.array(ind, dtype=bool)) for ind in new_pop]
        all_pop = population + new_pop
        all_fit = fitness + new_fit
        top = np.argsort(all_fit)[:pop_size]
        population = [all_pop[i] for i in top]
        fitness = [all_fit[i] for i in top]

        best_fit = min(fitness)
        fitness_history.append(best_fit)
        n_true_evals.append(pop_size)

    return np.array(population[0], dtype=bool), fitness_history, n_true_evals


naive_start = time.time()
naive_best_mask, naive_history, naive_true_counts = run_naive_ga(
    pop_size=40, n_generations=30, random_state=42
)
naive_time = time.time() - naive_start

# Validate both
clf_sa = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
sa_scores = cross_val_score(clf_sa, X[:, sa_best_mask], y, cv=CV5, scoring="accuracy")

clf_naive = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
naive_scores = cross_val_score(clf_naive, X[:, naive_best_mask], y, cv=CV5, scoring="accuracy")

# ── Summary comparison ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL COMPARISON — ALL EXERCISES")
print("=" * 70)

print(f"\n{'Method':<28} {'5-fold Acc':>12} {'Features':>10} {'Time (s)':>10} {'True Evals':>12}")
print("-" * 78)

methods_summary = [
    ("Baseline (all features)", BASELINE_ACC, baseline_scores.std(), N_FEATURES, None, None),
    ("ACO (elitist pheromone)", aco_scores.mean(), aco_scores.std(), int(aco_best_mask.sum()), aco_time, 40 * 20),
    ("Cooperative Co-evo (3 subpops)", cc_scores.mean(), cc_scores.std(), int(cc_best_mask.sum()), cc_time, None),
    ("Surrogate-Assisted GA", sa_scores.mean(), sa_scores.std(), int(sa_best_mask.sum()), sa_time, sum(sa_true_counts)),
    ("Naive GA (same gen)", naive_scores.mean(), naive_scores.std(), int(naive_best_mask.sum()), naive_time, sum(naive_true_counts)),
]

for name, acc, std, nf, t, n_evals in methods_summary:
    t_str = f"{t:.1f}" if t is not None else "N/A"
    evals_str = f"{n_evals}" if n_evals is not None else "N/A"
    print(f"{name:<28} {acc:>12.4f} {nf:>10} {t_str:>10} {evals_str:>12}")

print()

# Surrogate efficiency summary
total_sa_evals = sum(sa_true_counts)
total_naive_evals = sum(naive_true_counts)
reduction = (total_naive_evals - total_sa_evals) / total_naive_evals
print(f"Surrogate-Assisted GA efficiency:")
print(f"  True evaluations used: {total_sa_evals} vs naive {total_naive_evals}")
print(f"  Evaluation reduction: {reduction:.1%}")
print(f"  Accuracy trade-off: {sa_scores.mean() - naive_scores.mean():+.4f}")

print()
print("Self-check complete. All exercises produced results.")
print(f"Total cached evaluations across all exercises: {len(_eval_cache)}")

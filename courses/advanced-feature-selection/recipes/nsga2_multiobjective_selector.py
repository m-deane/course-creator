"""
nsga2_multiobjective_selector.py
---------------------------------
NSGA-II multi-objective feature selection using DEAP.

Objectives (both minimised):
  1. 1 - cross-validated accuracy     (maximise accuracy)
  2. n_selected / n_total             (minimise feature count ratio)

Steps:
  - Binary individual encoding (gene j = 1 means feature j is included)
  - NSGA-II selection with non-dominated sorting + crowding distance
  - Pareto front visualisation
  - Knee point detection via maximum curvature / minimum distance to ideal

Dependencies: deap (pip install deap) + standard data-science stack.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# DEAP imports
from deap import base, creator, tools, algorithms


# ---------------------------------------------------------------------------
# NSGA-II setup
# ---------------------------------------------------------------------------

def _setup_toolbox(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    cv: StratifiedKFold,
    estimator,
) -> base.Toolbox:
    """Register DEAP operators for NSGA-II."""

    # Create fitness and individual types (only once — guard against re-registration)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    def evaluate(individual: list[int]) -> tuple[float, float]:
        """Evaluate one NSGA-II individual; returns (1-accuracy, feature_ratio)."""
        mask = np.array(individual, dtype=bool)
        if mask.sum() == 0:
            # Penalise empty selection: worst accuracy, worst (maximum) feature ratio
            return (1.0, 1.0)
        X_sel = X[:, mask]
        scores = cross_val_score(estimator, X_sel, y, cv=cv, scoring="accuracy")
        acc = scores.mean()
        feature_ratio = mask.sum() / n_features
        return (1.0 - acc, feature_ratio)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


# ---------------------------------------------------------------------------
# Run NSGA-II
# ---------------------------------------------------------------------------

def nsga2_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 60,
    n_generations: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[list, list[tuple[float, float]]]:
    """Run NSGA-II and return the final Pareto front.

    Returns
    -------
    pareto_front : list of DEAP individuals on the Pareto front
    pareto_objectives : list of (1-accuracy, feature_ratio) tuples
    """
    np.random.seed(random_state)
    n_features = X.shape[1]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    estimator = DecisionTreeClassifier(max_depth=5, random_state=random_state)

    # selTournamentDCD requires pop_size divisible by 4
    pop_size = pop_size + (4 - pop_size % 4) % 4

    toolbox = _setup_toolbox(X, y, n_features, cv, estimator)
    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Assign crowding distance for initial population
    population = toolbox.select(population, len(population))

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)

    print(f"Running NSGA-II: {pop_size} individuals × {n_generations} generations …")
    for gen in range(n_generations):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if np.random.rand() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate only individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, pop_size)

        if (gen + 1) % 10 == 0:
            front0 = tools.sortNondominated(population, len(population), first_front_only=True)[0]
            obj_vals = [ind.fitness.values for ind in front0]
            best_acc = 1.0 - min(v[0] for v in obj_vals)
            print(f"  gen {gen + 1:>3}: Pareto front size = {len(front0)}, "
                  f"best accuracy = {best_acc:.4f}")

    # Extract final Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    pareto_objectives = [tuple(ind.fitness.values) for ind in pareto_front]
    return pareto_front, pareto_objectives


# ---------------------------------------------------------------------------
# Knee point detection
# ---------------------------------------------------------------------------

def detect_knee_point(objectives: list[tuple[float, float]]) -> int:
    """Find the knee point on the Pareto front (minimum distance to ideal point)."""
    obj_array = np.array(objectives)
    # Ideal point: minimum of each objective
    ideal = obj_array.min(axis=0)
    # Nadir point: maximum of each objective
    nadir = obj_array.max(axis=0)
    # Normalise to [0, 1]
    range_ = nadir - ideal + 1e-9
    normalised = (obj_array - ideal) / range_
    # Distance from normalised origin (ideal in normalised space = 0)
    distances = np.linalg.norm(normalised, axis=1)
    return int(np.argmin(distances))


# ---------------------------------------------------------------------------
# Pareto front plot
# ---------------------------------------------------------------------------

def plot_pareto_front(
    objectives: list[tuple[float, float]],
    knee_idx: int,
    save_path: str | None = None,
) -> None:
    """Plot the NSGA-II Pareto front with the knee point highlighted.

    Parameters
    ----------
    objectives : list[tuple[float, float]]
        Pareto front objective values as (1-accuracy, feature_ratio) tuples.
    knee_idx : int
        Index into objectives of the knee point solution.
    save_path : str, optional
        File path to save the figure. Displays interactively if None.
    """
    obj_array = np.array(objectives)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(obj_array[:, 1], 1 - obj_array[:, 0], color="steelblue",
               s=40, label="Pareto solutions")
    ax.scatter(obj_array[knee_idx, 1], 1 - obj_array[knee_idx, 0],
               color="red", s=120, zorder=5, label="Knee point")
    ax.set_xlabel("Feature ratio (smaller = fewer features)")
    ax.set_ylabel("CV Accuracy (higher = better)")
    ax.set_title("NSGA-II Pareto Front: Accuracy vs. Feature Count")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Pareto front plot saved to {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = load_breast_cancer()
    X_raw, y = data.data, data.target
    feature_names = np.array(data.feature_names)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    pareto_front, pareto_obj = nsga2_feature_selection(
        X, y, pop_size=50, n_generations=40, random_state=42
    )

    knee_idx = detect_knee_point(pareto_obj)
    knee_individual = pareto_front[knee_idx]
    knee_mask = np.array(knee_individual, dtype=bool)

    print(f"\nKnee point solution:")
    print(f"  Features selected : {knee_mask.sum()} / {len(knee_mask)}")
    err, feat_ratio = pareto_obj[knee_idx]
    print(f"  CV accuracy       : {1 - err:.4f}")
    print(f"  Feature ratio     : {feat_ratio:.4f}")
    print(f"  Feature names     : {list(feature_names[knee_mask])}")

    # Summary of all Pareto solutions
    summary = pd.DataFrame(
        [(1 - e, int(fr * len(knee_mask))) for e, fr in pareto_obj],
        columns=["cv_accuracy", "n_features"],
    ).sort_values("n_features")
    print("\nPareto front summary:")
    print(summary.to_string(index=False))

    plot_pareto_front(
        pareto_obj,
        knee_idx,
        save_path="/home/user/course-creator/courses/advanced-feature-selection/recipes/nsga2_pareto.png",
    )

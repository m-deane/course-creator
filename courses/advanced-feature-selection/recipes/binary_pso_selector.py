"""
binary_pso_selector.py
----------------------
Binary Particle Swarm Optimisation (BPSO) for feature selection.

Encoding: each particle is a binary vector of length p (1 = feature included).
Velocity update uses sigmoid transfer function to map real-valued velocity
to selection probability.  Global best (gbest) topology.

Fitness:  -accuracy_cv + lambda * (n_selected / p)    (minimise)
  - cross-validated accuracy rewards predictive power
  - parsimony penalty lambda controls sparsity

Reference: Kennedy & Eberhart 1997, "A Discrete Binary Version of the PSO".
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Sigmoid and velocity helpers
# ---------------------------------------------------------------------------

def _sigmoid(v: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))


def _velocity_clamp(v: np.ndarray, v_max: float) -> np.ndarray:
    return np.clip(v, -v_max, v_max)


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------

def _fitness(
    particle: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    estimator,
    cv: StratifiedKFold,
    parsimony: float,
) -> float:
    """Evaluate a binary feature mask.  Returns value to minimise."""
    selected = particle.astype(bool)
    if selected.sum() == 0:
        return 1.0  # worst possible — no features selected

    X_sel = X[:, selected]
    scores = cross_val_score(estimator, X_sel, y, cv=cv, scoring="accuracy")
    accuracy = scores.mean()
    sparsity_penalty = parsimony * selected.sum() / len(particle)
    return -accuracy + sparsity_penalty


# ---------------------------------------------------------------------------
# BPSO
# ---------------------------------------------------------------------------

def binary_pso_select(
    X: np.ndarray,
    y: np.ndarray,
    n_particles: int = 30,
    n_iterations: int = 50,
    w: float = 0.7,          # inertia weight
    c1: float = 1.5,          # cognitive coefficient
    c2: float = 1.5,          # social coefficient
    v_max: float = 4.0,        # velocity clamp
    parsimony: float = 0.1,   # sparsity penalty weight
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Run BPSO feature selection.

    Returns
    -------
    best_mask : boolean array of shape (n_features,)
    history : list of global best fitness per iteration
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    estimator = KNeighborsClassifier(n_neighbors=5)

    # Initialise positions (binary) and velocities (real)
    positions = rng.integers(0, 2, size=(n_particles, n_features)).astype(float)
    velocities = rng.uniform(-v_max, v_max, size=(n_particles, n_features))

    # Personal bests
    pbest_pos = positions.copy()
    pbest_fit = np.array([
        _fitness(positions[i], X, y, estimator, cv, parsimony)
        for i in range(n_particles)
    ])

    # Global best
    gbest_idx = int(np.argmin(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    history: list[float] = []

    for iteration in range(n_iterations):
        r1 = rng.uniform(0, 1, size=(n_particles, n_features))
        r2 = rng.uniform(0, 1, size=(n_particles, n_features))

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = _velocity_clamp(velocities, v_max)

        # Position update via sigmoid transfer
        transfer_prob = _sigmoid(velocities)
        rand_matrix = rng.uniform(0, 1, size=(n_particles, n_features))
        positions = (rand_matrix < transfer_prob).astype(float)

        # Evaluate and update personal / global bests
        for i in range(n_particles):
            fit = _fitness(positions[i], X, y, estimator, cv, parsimony)
            if fit < pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest_pos = positions[i].copy()

        history.append(gbest_fit)
        if (iteration + 1) % 10 == 0:
            n_sel = int(gbest_pos.sum())
            print(f"  iter {iteration + 1:>3}: gbest_fitness = {gbest_fit:.4f}  "
                  f"(n_features = {n_sel})")

    return gbest_pos.astype(bool), history


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(history: list[float], save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, linewidth=2, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Global best fitness (lower = better)")
    ax.set_title("BPSO Convergence")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Convergence plot saved to {save_path}")
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

    n_particles = 20
    print("Running Binary PSO feature selection …")
    print(f"  {X.shape[1]} features, {n_particles} particles, 40 iterations\n")

    best_mask, history = binary_pso_select(
        X, y,
        n_particles=n_particles,
        n_iterations=40,
        parsimony=0.05,
        random_state=42,
    )

    selected_names = feature_names[best_mask]
    print(f"\nSelected {best_mask.sum()} features:")
    for name in selected_names:
        print(f"  - {name}")

    # Report full-data accuracy with selected subset
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    acc_all = cross_val_score(knn, X, y, cv=5, scoring="accuracy").mean()
    acc_sel = cross_val_score(knn, X[:, best_mask], y, cv=5, scoring="accuracy").mean()
    print(f"\nCV accuracy — all features:      {acc_all:.4f}")
    print(f"CV accuracy — BPSO selection:    {acc_sel:.4f}")
    print(f"Feature reduction: {X.shape[1]} → {best_mask.sum()}")

    plot_convergence(
        history,
        save_path="/home/user/course-creator/courses/advanced-feature-selection/recipes/bpso_convergence.png",
    )

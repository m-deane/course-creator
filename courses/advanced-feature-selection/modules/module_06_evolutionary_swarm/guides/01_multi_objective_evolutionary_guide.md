# Multi-Objective Evolutionary Feature Selection

## The Accuracy-Complexity-Stability Trilemma

Single-objective feature selection reduces the problem to a scalar: minimize (or maximize) one number. This forces you to commit to a trade-off weight before you understand the trade-off surface. Multi-objective evolutionary algorithms (MOEAs) defer that commitment by returning a **Pareto front** — the complete set of non-dominated solutions. A decision-maker, not an arbitrary weight, then picks the preferred operating point.

For feature selection, three objectives compete:

| Objective | Minimise | Why It Matters |
|---|---|---|
| **Prediction error** | Classification error or MSE | Primary model utility |
| **Feature count** | $\|\mathbf{x}\|_0$ | Inference cost, interpretability, data collection |
| **Instability** | Jaccard distance across CV folds | Reproducibility, trust |

Adding stability as a third objective converts a bi-objective problem into a **many-objective** problem. NSGA-II handles two to three objectives well; NSGA-III and MOEA/D were designed for four or more.

---

## NSGA-II: Non-Dominated Sorting Genetic Algorithm II

Reference: Deb, Pratap, Agarwal & Meyarivan (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

### Core Concepts

**Domination**: Solution $\mathbf{a}$ dominates $\mathbf{b}$ (written $\mathbf{a} \prec \mathbf{b}$) if $\mathbf{a}$ is no worse on all objectives and strictly better on at least one.

$$\mathbf{a} \prec \mathbf{b} \iff \forall k: f_k(\mathbf{a}) \leq f_k(\mathbf{b}) \;\wedge\; \exists k: f_k(\mathbf{a}) < f_k(\mathbf{b})$$

**Non-dominated front (rank 1)**: solutions not dominated by any other solution in the population.

**Rank $i$ front**: solutions dominated only by solutions in fronts 1 through $i-1$.

### Non-Dominated Sorting

The classic $O(MN^2)$ sorting algorithm (Deb 2002, Algorithm 1):

```
For each individual p:
    S_p = {}  (set dominated by p)
    n_p = 0   (domination counter)
    For each individual q:
        if p dominates q:  add q to S_p
        elif q dominates p: n_p += 1
    if n_p == 0: p is in front F_1

For each front F_i:
    For each p in F_i:
        For each q in S_p:
            n_q -= 1
            if n_q == 0: q goes to F_{i+1}
```

Practical implementation uses numpy broadcasting for vectorised domination checks.

### Crowding Distance

Within a front, crowding distance measures how isolated a solution is. A large crowding distance indicates a solution occupies a sparse region of objective space — desirable for diversity.

For objective $k$, sort the front by $f_k$. The crowding distance contribution from $k$ for solution $i$ is:

$$d_i^k = \frac{f_k(i+1) - f_k(i-1)}{f_k^{\max} - f_k^{\min}}$$

Boundary solutions receive $\infty$. Total crowding distance: $d_i = \sum_k d_i^k$.

### Binary Tournament with Dominance

Selection operator: pick 2 random candidates, return the winner by:
1. Lower rank wins
2. Tie on rank → higher crowding distance wins

### NSGA-II Main Loop

```
Initialise population P_0 of size N
Evaluate all individuals (both objectives)
Non-dominated sort P_0 into fronts F_1, F_2, ...

For each generation t:
    Q_t = generate offspring from P_t (tournament, crossover, mutation)
    R_t = P_t ∪ Q_t  (size 2N)
    Sort R_t → fronts F_1, F_2, ...
    P_{t+1} = {}
    i = 1
    While |P_{t+1}| + |F_i| <= N:
        Assign crowding distance to F_i
        P_{t+1} = P_{t+1} ∪ F_i
        i += 1
    Sort remaining front F_i by crowding distance (descending)
    Fill P_{t+1} with best crowded solutions until |P_{t+1}| = N
```

### Full NSGA-II Implementation with DEAP

```python
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ── Dataset ──────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
N_FEATURES = X_scaled.shape[1]

# ── DEAP setup ────────────────────────────────────────────────────────────────
# Two objectives, both minimised: error_rate, feature_fraction
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_bool,
    n=N_FEATURES,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ── Fitness function ──────────────────────────────────────────────────────────
def evaluate_individual(individual):
    mask = np.array(individual, dtype=bool)
    n_selected = mask.sum()

    if n_selected == 0:
        return 1.0, 1.0  # worst possible

    X_sub = X_scaled[:, mask]
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_sub, y, cv=cv, scoring="accuracy")
    error_rate = 1.0 - scores.mean()
    feature_fraction = n_selected / N_FEATURES
    return error_rate, feature_fraction

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / N_FEATURES)
toolbox.register("select", tools.selNSGA2)

# ── Run NSGA-II ────────────────────────────────────────────────────────────────
POPULATION_SIZE = 80
N_GENERATIONS = 40
CXPB, MUTPB = 0.7, 0.2

random.seed(42)
np.random.seed(42)

population = toolbox.population(n=POPULATION_SIZE)

# Evaluate initial population
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Assign crowding distance to initial population
population = toolbox.select(population, len(population))

logbook = tools.Logbook()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)
stats.register("mean", np.mean, axis=0)

for gen in range(N_GENERATIONS):
    offspring = tools.selTournamentDCD(population, len(population))
    offspring = [toolbox.clone(ind) for ind in offspring]

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate individuals with invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select next generation (NSGA-II selection)
    population = toolbox.select(population + offspring, POPULATION_SIZE)

    record = stats.compile(population)
    logbook.record(gen=gen, **record)
    if gen % 10 == 0:
        print(f"Gen {gen:3d} | min_error={record['min'][0]:.4f}, "
              f"min_features={record['min'][1]:.4f}")

# ── Extract Pareto front ───────────────────────────────────────────────────────
pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
print(f"\nPareto front size: {len(pareto_front)}")
```

---

## Pareto Front Analysis

### Ideal and Nadir Points

- **Ideal point** $\mathbf{z}^*$: minimum of each objective independently (not always achievable simultaneously).
- **Nadir point** $\mathbf{z}^{\text{nad}}$: maximum of each objective over the Pareto front (worst acceptable values).

These define the bounding box of the Pareto front and are used for normalisation in NSGA-III and MOEA/D.

### Knee Point Detection

The knee point is the solution with the maximum perpendicular distance from the line connecting the extreme Pareto front solutions. It represents the point of diminishing returns — adding more features yields little accuracy improvement.

```python
import numpy as np

def find_knee_point(pareto_objectives: np.ndarray) -> int:
    """
    Find knee point via maximum perpendicular distance from the line
    connecting extreme solutions on the Pareto front.

    Parameters
    ----------
    pareto_objectives : ndarray of shape (n_solutions, 2)
        Objective values [error_rate, feature_fraction], both minimised.

    Returns
    -------
    int
        Index of the knee point solution.
    """
    # Sort by first objective
    sorted_idx = np.argsort(pareto_objectives[:, 0])
    pts = pareto_objectives[sorted_idx]

    # Line from first to last point
    start, end = pts[0], pts[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        return sorted_idx[0]

    # Perpendicular distance for each point
    distances = []
    for pt in pts:
        vec = pt - start
        proj = np.dot(vec, line_vec) / line_len
        proj_pt = start + proj * line_vec / line_len
        dist = np.linalg.norm(pt - proj_pt)
        distances.append(dist)

    knee_idx_in_sorted = np.argmax(distances)
    return sorted_idx[knee_idx_in_sorted]


def find_knee_point_marginal_utility(
    pareto_objectives: np.ndarray,
    accuracy_weight: float = 0.7,
    complexity_weight: float = 0.3,
) -> int:
    """
    Select the solution that maximises a weighted utility function.
    Appropriate when domain knowledge specifies relative importance.
    """
    # Normalise objectives to [0, 1]
    mins = pareto_objectives.min(axis=0)
    maxs = pareto_objectives.max(axis=0)
    ranges = maxs - mins
    ranges[ranges < 1e-12] = 1.0
    normalised = (pareto_objectives - mins) / ranges

    # Utility: maximise accuracy (minimise error) and minimise features
    utility = -accuracy_weight * normalised[:, 0] - complexity_weight * normalised[:, 1]
    return int(np.argmax(utility))
```

### Selecting a Single Solution

Three principled strategies:

1. **Knee point**: automatic, no parameters required. Best when the trade-off surface is convex with a clear inflection.
2. **Marginal utility / weighted sum of normalised objectives**: requires specifying weights, but allows domain knowledge injection. Use when stakeholders can express relative importance.
3. **Domain constraints**: filter solutions satisfying hard constraints (e.g., "at most 10 features"), then pick best accuracy within the feasible set.

```python
def select_by_constraint(pareto_front, max_features: int):
    """Select best-accuracy solution using at most max_features."""
    feasible = [
        ind for ind in pareto_front
        if sum(ind) <= max_features
    ]
    if not feasible:
        # Relax: pick smallest feature count
        return min(pareto_front, key=lambda ind: sum(ind))
    return min(feasible, key=lambda ind: ind.fitness.values[0])
```

---

## NSGA-III: Reference Point-Based Many-Objective Selection

Reference: Deb & Jain (2014). *An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach*. IEEE TEC, 18(4), 577–601.

NSGA-III replaces crowding distance with structured **reference points** uniformly spread on a unit hyperplane. This maintains diversity in objective spaces with four or more dimensions where crowding distance degrades.

**Reference point generation** (Das & Dennis, 1998):
For $M$ objectives and $H$ divisions per objective, the number of reference points is $\binom{H+M-1}{M-1}$.

**Selection procedure** (replacing crowding distance):
1. Associate each population member with its nearest reference point (after normalisation to the ideal-nadir hyperplane).
2. Compute the **niche count** $\rho_j$ = number of already-selected members near reference point $j$.
3. Among solutions on the boundary front, prefer solutions near under-populated reference points.

NSGA-III is the method of choice when optimising accuracy + feature count + stability + computational cost (four objectives).

---

## MOEA/D: Decomposition-Based Multi-Objective Optimisation

Reference: Zhang & Li (2007). *MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition*. IEEE TEC, 11(6), 712–731.

MOEA/D decomposes the multi-objective problem into $N$ scalar subproblems using a set of weight vectors $\{\boldsymbol{\lambda}^1, \ldots, \boldsymbol{\lambda}^N\}$. Each subproblem $i$ uses the **Tchebycheff decomposition**:

$$g^{\text{tch}}(\mathbf{x} | \boldsymbol{\lambda}^i, \mathbf{z}^*) = \max_{k=1}^M \lambda_k^i |f_k(\mathbf{x}) - z_k^*|$$

**MOEA/D algorithm**:
1. Generate $N$ uniformly distributed weight vectors; define $T$ nearest neighbours for each vector.
2. Each subproblem $i$ maintains one solution $\mathbf{x}^i$.
3. For each subproblem: select two parents from its neighbourhood, apply crossover/mutation, update the neighbourhood if the offspring improves any neighbour's scalar objective.

MOEA/D is computationally cheaper than NSGA-II on large populations because updates are local. It excels when the Pareto front is simple (e.g., concave or linear); NSGA-II is more robust on disconnected fronts.

---

## Algorithm Comparison

| Algorithm | Objectives | Diversity Mechanism | Complexity | Best Use Case |
|---|---|---|---|---|
| NSGA-II | 2–3 | Crowding distance | $O(MN^2)$ | Feature count vs. accuracy |
| NSGA-III | 3–15 | Reference points | $O(MN^2)$ | Many-objective feature selection |
| MOEA/D | 2–20 | Weight vector neighbourhood | $O(MN \cdot T)$ | Decomposable problems |
| SPEA2 | 2–4 | Archive + density | $O(MN^2)$ | Small populations |

---

## Key Takeaways

- Multi-objective optimisation returns a **Pareto front**, not a single answer. The decision-maker selects from this front.
- NSGA-II is the standard baseline for two- and three-objective feature selection. DEAP implements it with `tools.selNSGA2` and `tools.selTournamentDCD`.
- Knee point detection is a principled, parameter-free way to auto-select from the Pareto front.
- NSGA-III and MOEA/D extend to many objectives at the cost of additional configuration (reference points, weight vectors).
- Stability as a third objective is practically important: a feature set that changes across resamples is not deployable.

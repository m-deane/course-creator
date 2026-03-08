# Swarm Intelligence for Feature Selection

## Overview

Swarm intelligence algorithms draw inspiration from collective natural behaviour — bird flocking (PSO), differential vector mutation (DE), ant pheromone trails (ACO). All three operate on a population of candidate feature subsets and use different update rules to move that population toward better regions of the search space. This guide covers the mathematics, implementation details, and practical trade-offs for each.

---

## Particle Swarm Optimisation (PSO)

Reference: Kennedy & Eberhart (1995). *Particle swarm optimization*. Proceedings of ICNN, 4, 1942–1948.

### Core PSO Mechanics

Each particle $i$ has a position $\mathbf{x}_i \in \mathbb{R}^D$ (the candidate solution) and a velocity $\mathbf{v}_i \in \mathbb{R}^D$. Update equations:

$$\mathbf{v}_i^{t+1} = w \mathbf{v}_i^t + c_1 r_1 (\mathbf{p}_i - \mathbf{x}_i^t) + c_2 r_2 (\mathbf{g} - \mathbf{x}_i^t)$$

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t + \mathbf{v}_i^{t+1}$$

Where:
- $w$: inertia weight (typically $0.4$–$0.9$, often linearly decayed)
- $c_1$: cognitive coefficient (attraction to personal best $\mathbf{p}_i$, typically $2.0$)
- $c_2$: social coefficient (attraction to global best $\mathbf{g}$, typically $2.0$)
- $r_1, r_2 \sim U(0,1)$: random scaling factors, sampled per dimension per iteration

### Binary PSO for Feature Selection

Feature selection requires binary decisions ($0$ = exclude, $1$ = include). Continuous PSO is adapted to binary space using a **sigmoid transfer function**:

$$S(v_{id}) = \frac{1}{1 + e^{-v_{id}}}$$

$$x_{id}^{t+1} = \begin{cases} 1 & \text{if } U(0,1) < S(v_{id}^{t+1}) \\ 0 & \text{otherwise} \end{cases}$$

**Velocity clamping**: prevents sigmoid saturation. A clamped velocity $v_{\max}$ keeps $S(v)$ in a range where both $0$ and $1$ are reachable:

$$v_{id} \leftarrow \text{clip}(v_{id}, -v_{\max}, v_{\max})$$

Typical choice: $v_{\max} = 4.0$, giving $S(\pm4) \approx 0.018 / 0.982$.

### Continuous-to-Binary Transfer Functions

Four common mappings from continuous velocity to bit-flip probability:

| Transfer Function | Formula | Notes |
|---|---|---|
| **S-shaped** | $S(v) = \frac{1}{1+e^{-v}}$ | Original Kennedy & Eberhart |
| **V-shaped** | $V(v) = \left\|\tanh(v)\right\|$ | Maintains sign information |
| **Threshold** | $x = 1$ if $x_{\text{cont}} > 0.5$ | Deterministic after normalisation |
| **Probabilistic** | $P(\text{flip}) = S(|v|) \cdot \text{sign}$ | Asymmetric flipping |

V-shaped transfer functions (Mirjalili 2014) often outperform S-shaped for feature selection by preserving directional velocity information.

### Binary PSO Implementation

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ── Dataset ────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)
N_FEATURES = X_scaled.shape[1]

# ── PSO Parameters ─────────────────────────────────────────────────────────────
N_PARTICLES = 40
N_ITERATIONS = 50
W_MAX, W_MIN = 0.9, 0.4       # inertia decay range
C1, C2 = 2.0, 2.0             # cognitive and social coefficients
V_MAX = 4.0                   # velocity clamp

def sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))

def evaluate_subset(mask: np.ndarray) -> float:
    """Return cross-validation error rate for the given binary mask."""
    if mask.sum() == 0:
        return 1.0
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled[:, mask], y, cv=cv, scoring="accuracy")
    return 1.0 - scores.mean()

rng = np.random.default_rng(42)

# ── Initialise ─────────────────────────────────────────────────────────────────
positions = rng.integers(0, 2, size=(N_PARTICLES, N_FEATURES)).astype(float)
velocities = rng.uniform(-V_MAX, V_MAX, size=(N_PARTICLES, N_FEATURES))

# Evaluate initial positions
fitness = np.array([evaluate_subset(p.astype(bool)) for p in positions])

personal_best_pos = positions.copy()
personal_best_fit = fitness.copy()

global_best_idx = np.argmin(fitness)
global_best_pos = positions[global_best_idx].copy()
global_best_fit = fitness[global_best_idx]

history = [global_best_fit]

# ── Main Loop ──────────────────────────────────────────────────────────────────
for t in range(N_ITERATIONS):
    # Linearly decaying inertia
    w = W_MAX - (W_MAX - W_MIN) * t / N_ITERATIONS

    r1 = rng.random(size=(N_PARTICLES, N_FEATURES))
    r2 = rng.random(size=(N_PARTICLES, N_FEATURES))

    # Velocity update
    velocities = (
        w * velocities
        + C1 * r1 * (personal_best_pos - positions)
        + C2 * r2 * (global_best_pos - positions)
    )
    # Velocity clamping
    velocities = np.clip(velocities, -V_MAX, V_MAX)

    # Binary position update via sigmoid transfer
    probs = sigmoid(velocities)
    positions = (rng.random(size=(N_PARTICLES, N_FEATURES)) < probs).astype(float)

    # Evaluate
    fitness = np.array([evaluate_subset(p.astype(bool)) for p in positions])

    # Update personal bests
    improved = fitness < personal_best_fit
    personal_best_pos[improved] = positions[improved].copy()
    personal_best_fit[improved] = fitness[improved]

    # Update global best
    current_best_idx = np.argmin(personal_best_fit)
    if personal_best_fit[current_best_idx] < global_best_fit:
        global_best_pos = personal_best_pos[current_best_idx].copy()
        global_best_fit = personal_best_fit[current_best_idx]

    history.append(global_best_fit)
    if t % 10 == 0:
        n_selected = global_best_pos.sum()
        print(f"Iter {t:3d} | best_error={global_best_fit:.4f}, "
              f"n_features={n_selected:.0f}")

print(f"\nFinal: error={global_best_fit:.4f}, "
      f"features={global_best_pos.sum():.0f}/{N_FEATURES}")
```

---

## PSO Topology

The topology determines which particles can influence each other. The social component $c_2 r_2 (\mathbf{g} - \mathbf{x}_i)$ uses a **best** that depends on topology:

| Topology | Description | Convergence Speed | Exploration |
|---|---|---|---|
| **Global best (gbest)** | All particles share one global best | Fast | Low |
| **Local best (lbest)** | Ring topology; each particle sees $k$ neighbours | Slow | High |
| **Ring** | Each particle sees 2 neighbours (left/right) | Slowest | Highest |
| **Von Neumann** | Grid topology; each sees 4 neighbours | Medium | Medium-high |

**Global best** is the default and fastest-converging. **Local best with ring** is more robust on multimodal problems (multiple local optima in feature space).

```python
def ring_best(personal_bests: np.ndarray, personal_fits: np.ndarray,
              i: int, k: int = 2) -> np.ndarray:
    """Return best position in the k-neighbourhood of particle i (ring)."""
    n = len(personal_fits)
    neighbours = [(i + j) % n for j in range(-k, k + 1)]
    best_neighbour = min(neighbours, key=lambda idx: personal_fits[idx])
    return personal_bests[best_neighbour]
```

---

## Differential Evolution (DE)

Reference: Storn & Price (1997). *Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*. Journal of Global Optimization, 11(4), 341–359.

### DE Mutation Strategies

DE operates on continuous vectors but can be adapted to binary feature selection. The three standard strategies:

**DE/rand/1** (most common, best exploration):
$$\mathbf{v}_i = \mathbf{x}_{r_1} + F \cdot (\mathbf{x}_{r_2} - \mathbf{x}_{r_3})$$

**DE/best/1** (fastest convergence, higher exploitation):
$$\mathbf{v}_i = \mathbf{x}_{\text{best}} + F \cdot (\mathbf{x}_{r_1} - \mathbf{x}_{r_2})$$

**DE/current-to-best/1** (balance of both):
$$\mathbf{v}_i = \mathbf{x}_i + F \cdot (\mathbf{x}_{\text{best}} - \mathbf{x}_i) + F \cdot (\mathbf{x}_{r_1} - \mathbf{x}_{r_2})$$

Where $F \in [0, 2]$ is the mutation scale factor (typically $0.5$–$0.9$) and $r_1, r_2, r_3$ are distinct random indices different from $i$.

### Binomial Crossover

After mutation, each trial vector dimension is accepted with probability $CR$ (crossover rate, typically $0.7$–$0.9$):

$$u_{ij} = \begin{cases} v_{ij} & \text{if } U(0,1) \leq CR \text{ or } j = j_{\text{rand}} \\ x_{ij} & \text{otherwise} \end{cases}$$

$j_{\text{rand}}$ ensures at least one dimension from the mutant is used.

### DE for Binary Feature Selection

Continuous DE vectors are mapped to binary using a threshold after sigmoid transformation:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)
N_FEATURES = X_scaled.shape[1]

# ── DE Parameters ──────────────────────────────────────────────────────────────
POP_SIZE = 40
N_GENERATIONS = 50
F = 0.5          # mutation scale factor
CR = 0.7         # crossover rate

def evaluate_continuous(x: np.ndarray) -> float:
    """Convert continuous DE vector to binary mask, evaluate."""
    mask = (1.0 / (1.0 + np.exp(-x))) > 0.5
    if not mask.any():
        mask[np.argmax(x)] = True  # force at least one feature
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        clf, X_scaled[:, mask], y, cv=cv, scoring="accuracy"
    )
    return 1.0 - scores.mean()

rng = np.random.default_rng(42)

# Initialise population in continuous space [-3, 3] (maps to [0.05, 0.95] after sigmoid)
population = rng.uniform(-3, 3, size=(POP_SIZE, N_FEATURES))
fitness = np.array([evaluate_continuous(ind) for ind in population])

best_idx = np.argmin(fitness)
best_fitness_history = [fitness[best_idx]]

for gen in range(N_GENERATIONS):
    for i in range(POP_SIZE):
        # Select three distinct random indices ≠ i
        candidates = [j for j in range(POP_SIZE) if j != i]
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

        # DE/rand/1 mutation
        mutant = population[r1] + F * (population[r2] - population[r3])

        # Binomial crossover
        j_rand = rng.integers(N_FEATURES)
        cross_mask = (rng.random(N_FEATURES) <= CR)
        cross_mask[j_rand] = True
        trial = np.where(cross_mask, mutant, population[i])

        # Greedy selection
        trial_fit = evaluate_continuous(trial)
        if trial_fit < fitness[i]:
            population[i] = trial
            fitness[i] = trial_fit

    current_best = np.argmin(fitness)
    best_fitness_history.append(fitness[current_best])
    if gen % 10 == 0:
        best_mask = (1.0 / (1.0 + np.exp(-population[current_best]))) > 0.5
        print(f"Gen {gen:3d} | error={fitness[current_best]:.4f}, "
              f"features={best_mask.sum()}")
```

### Self-Adaptive DE Variants

**jDE** (Brest et al., 2006): each individual carries its own $F_i$ and $CR_i$, updated by a self-adaptation rule that favours parameter values that have produced improvements.

**SHADE** (Tanabe & Fukunaga, 2013): maintains a historical archive of successful $F$ and $CR$ values. New parameters are sampled from a Cauchy/Gaussian centred on the historical means.

**L-SHADE**: extends SHADE with a linear population size reduction — the population shrinks from $N_{\max}$ to $N_{\min}$ over generations, concentrating resources on the best solutions.

---

## Ant Colony Optimisation (ACO)

### Graph Construction for Feature Selection

ACO solves feature selection by treating it as a path through a feature graph:

- **Nodes**: features $\{f_1, \ldots, f_D\}$ plus a START node
- **Pheromone** $\tau_{ij}$: accumulated evidence for including feature $j$ after feature $i$
- **Heuristic** $\eta_j$: static feature quality (e.g., mutual information score)

### Ant Construction

Each ant builds a feature subset by probabilistically selecting features:

$$P(j | i, \text{visited}) = \frac{\tau_{ij}^\alpha \cdot \eta_j^\beta}{\sum_{k \notin \text{visited}} \tau_{ik}^\alpha \cdot \eta_k^\beta}$$

Parameters: $\alpha$ (pheromone weight), $\beta$ (heuristic weight). Selection continues until a termination criterion is met (e.g., fixed $k$ features, or no further improvement).

### Pheromone Update Rules

**Evaporation** (applied each iteration):
$$\tau_{ij} \leftarrow (1 - \rho) \cdot \tau_{ij}$$

$\rho \in (0, 1)$ is the evaporation rate (typically $0.1$–$0.5$). Evaporation prevents pheromone saturation and allows ACO to forget poor solutions.

**Deposit** (by ants that used edge $(i,j)$):
$$\tau_{ij} \leftarrow \tau_{ij} + \sum_{k : (i,j) \in S_k} Q / L_k$$

$Q$ is a constant, $L_k$ is the objective value of ant $k$'s solution. Better solutions deposit more pheromone.

**Elite ants** (ACO-elite): the best-so-far solution deposits additional pheromone with weight $e$, strongly reinforcing the best path found.

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def compute_heuristic(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mutual information as heuristic for feature quality."""
    mi = mutual_info_classif(X, y, random_state=42)
    mi_norm = mi / (mi.sum() + 1e-12)
    return mi_norm + 0.01  # avoid zero heuristic

def aco_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features_select: int = 10,
    n_ants: int = 20,
    n_iterations: int = 50,
    alpha: float = 1.0,      # pheromone weight
    beta: float = 2.0,       # heuristic weight
    rho: float = 0.1,        # evaporation rate
    q: float = 1.0,          # pheromone deposit constant
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    D = X.shape[1]
    heuristic = compute_heuristic(X, y)
    pheromone = np.ones(D)  # initialise uniformly

    def build_solution(pheromone: np.ndarray) -> np.ndarray:
        available = list(range(D))
        selected = []
        for _ in range(n_features_select):
            ph_h = (pheromone[available] ** alpha) * (heuristic[available] ** beta)
            probs = ph_h / ph_h.sum()
            chosen = rng.choice(available, p=probs)
            selected.append(chosen)
            available.remove(chosen)
        return np.array(selected)

    def evaluate_mask(indices: np.ndarray) -> float:
        mask = np.zeros(D, dtype=bool)
        mask[indices] = True
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X[:, mask], y, cv=cv, scoring="accuracy")
        return 1.0 - scores.mean()

    best_mask = None
    best_fitness = np.inf

    for iteration in range(n_iterations):
        ant_solutions = [build_solution(pheromone) for _ in range(n_ants)]
        ant_fitness = [evaluate_mask(sol) for sol in ant_solutions]

        # Evaporation
        pheromone *= (1 - rho)

        # Deposit
        for sol, fit in zip(ant_solutions, ant_fitness):
            deposit = q / (fit + 1e-12)
            pheromone[sol] += deposit

        # Update best
        best_ant_idx = np.argmin(ant_fitness)
        if ant_fitness[best_ant_idx] < best_fitness:
            best_fitness = ant_fitness[best_ant_idx]
            best_mask = ant_solutions[best_ant_idx].copy()

        if iteration % 10 == 0:
            print(f"Iter {iteration:3d} | best_error={best_fitness:.4f}")

    final_mask = np.zeros(D, dtype=bool)
    final_mask[best_mask] = True
    return final_mask
```

---

## Algorithm Comparison: PSO vs DE vs ACO vs GA

| Criterion | PSO | DE | ACO | GA |
|---|---|---|---|---|
| **Representation** | Continuous + sigmoid | Continuous + threshold | Discrete path | Binary string |
| **Population interaction** | Global/local best | Difference vectors | Pheromone matrix | Parent pairs |
| **Mutation** | Velocity update | Differential mutation | Probabilistic construction | Bit-flip |
| **Recombination** | None (implicit) | Binomial crossover | None | One-point, two-point |
| **Memory** | Personal best | None (stateless) | Pheromone (shared memory) | None |
| **Parameter sensitivity** | $w, c_1, c_2, v_{\max}$ | $F, CR$ | $\alpha, \beta, \rho, k$ | $p_c, p_m$ |
| **Best for** | Continuous + binary | Continuous spaces | Discrete, sequential | Binary, combinatorial |
| **Convergence speed** | Fast | Medium | Slow (builds paths) | Medium |
| **Exploration** | Medium | High | High | Medium |
| **Computational cost** | Low | Medium | High (path construction) | Low |
| **Parallelism** | Trivial | Trivial | Sequential construction | Trivial |

### Practical Recommendations

- **PSO**: use when you need fast convergence and are comfortable with the sigmoid transfer function. Binary PSO with velocity clamping is competitive with GA on most benchmark datasets.
- **DE**: use when the feature space is moderate-sized ($D < 200$) and you want robust convergence without hyperparameter sensitivity. DE/rand/1 is the safest default.
- **ACO**: use when features have sequential or dependency structure (e.g., time-series lag features, NLP n-gram features). ACO's path construction naturally captures ordered dependencies. Computationally expensive.
- **GA**: use as the baseline. Well-understood, many implementations, easily combined with domain knowledge through custom crossover and mutation operators.

---

## Key Takeaways

- Binary PSO adapts continuous PSO to feature selection via sigmoid transfer functions. Velocity clamping ($v_{\max} = 4.0$) prevents premature convergence.
- DE's differential mutation (`DE/rand/1`) is parameter-robust. $F = 0.5$, $CR = 0.7$ are good defaults.
- Self-adaptive DE variants (SHADE, L-SHADE) eliminate manual parameter tuning and are the current state-of-the-art for continuous optimisation.
- ACO models feature selection as path construction through a feature graph. Pheromone is shared memory that accumulates evidence for good feature combinations.
- No single algorithm dominates: runtime budget, feature space structure, and required diversity all influence the choice.

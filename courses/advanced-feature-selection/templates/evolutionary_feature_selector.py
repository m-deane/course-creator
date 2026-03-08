"""
evolutionary_feature_selector.py
---------------------------------
Unified evolutionary feature selection engine.

Supported algorithms (all share one interface)
-----------------------------------------------
- GA   : Genetic Algorithm with tournament selection, crossover, mutation
- BPSO : Binary Particle Swarm Optimisation
- DE   : Differential Evolution with binary encoding

Multi-objective mode (accuracy vs feature count) uses a Pareto front to
return a trade-off curve; single-objective mode returns the best solution.

Usage
-----
    from evolutionary_feature_selector import EvoConfig, EvolutionaryFeatureSelector

    cfg = EvoConfig(
        algorithm="GA",
        n_generations=50,
        population_size=40,
        multi_objective=True,
        task="classification",
    )
    evo = EvolutionaryFeatureSelector(cfg)
    evo.fit(X_train, y_train)

    # Single best mask (single-objective) or Pareto front head
    mask = evo.best_mask_
    X_sel = X_train[:, mask]

    # Full Pareto front (multi-objective)
    for ind in evo.pareto_front_:
        print(ind.n_features, ind.fitness)
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FEATURES_IN_MASK = 1          # chromosome must keep at least 1 feature
DIVERSITY_HAMMING_THRESHOLD = 0.05  # flag low diversity when mean pair dist < this
FITNESS_CACHE_MAX_SIZE = 4096


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EvoConfig:
    """Configuration for the evolutionary feature selector.

    Parameters
    ----------
    algorithm : str
        One of ``{"GA", "BPSO", "DE"}``.
    population_size : int
        Number of individuals per generation.
    n_generations : int
        Maximum number of generations / iterations.
    task : str
        ``"classification"`` or ``"regression"``.
    cv_folds : int
        Cross-validation folds used to evaluate each individual.
    multi_objective : bool
        If True, optimise (fitness, -n_features) jointly via Pareto fronts.
    alpha : float
        Weight of feature-count penalty in single-objective mode.
        Fitness = accuracy - alpha * (n_selected / n_total).
    crossover_prob : float
        GA crossover probability.
    mutation_prob : float
        GA/DE bit-flip mutation probability.
    ga_tournament_size : int
        Tournament size for GA selection.
    pso_inertia : float
        PSO inertia weight.
    pso_c1 : float
        PSO cognitive coefficient.
    pso_c2 : float
        PSO social coefficient.
    de_f : float
        DE differential weight (scale factor).
    de_cr : float
        DE crossover rate.
    convergence_patience : int
        Stop early if best fitness doesn't improve for this many generations.
    random_state : int
        Reproducibility seed.
    n_jobs : int
        Joblib workers (passed to sklearn estimators).
    """

    algorithm: str = "GA"
    population_size: int = 40
    n_generations: int = 50
    task: str = "classification"
    cv_folds: int = 3
    multi_objective: bool = False
    alpha: float = 0.01
    crossover_prob: float = 0.8
    mutation_prob: float = 0.02
    ga_tournament_size: int = 3
    pso_inertia: float = 0.7
    pso_c1: float = 1.5
    pso_c2: float = 1.5
    de_f: float = 0.8
    de_cr: float = 0.9
    convergence_patience: int = 10
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self) -> None:
        valid_algos = {"GA", "BPSO", "DE"}
        if self.algorithm not in valid_algos:
            raise ValueError(f"algorithm must be one of {valid_algos}")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")


# ---------------------------------------------------------------------------
# Individual / chromosome
# ---------------------------------------------------------------------------
@dataclass
class Individual:
    """One individual in the population (binary feature mask).

    Attributes
    ----------
    mask : np.ndarray
        Binary vector of length n_features.
    fitness : float
        Primary objective (higher is better).
    n_features : int
        Number of selected features.
    """

    mask: np.ndarray
    fitness: float = -np.inf
    n_features: int = 0

    def __post_init__(self) -> None:
        self.n_features = int(self.mask.sum())

    def clone(self) -> "Individual":
        return Individual(mask=self.mask.copy(), fitness=self.fitness, n_features=self.n_features)

    def dominates(self, other: "Individual") -> bool:
        """Pareto dominance: self is better on both objectives."""
        return self.fitness >= other.fitness and self.n_features <= other.n_features and (
            self.fitness > other.fitness or self.n_features < other.n_features
        )


# ---------------------------------------------------------------------------
# Fitness evaluator
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    """CV-based fitness evaluation with result caching.

    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    config : EvoConfig
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, config: EvoConfig) -> None:
        self.X = X
        self.y = y
        self.cfg = config
        self._cache: Dict[bytes, float] = {}
        self._n_evaluations = 0

    def evaluate(self, mask: np.ndarray) -> float:
        """Return CV fitness for a binary mask.

        Parameters
        ----------
        mask : np.ndarray of dtype bool or int

        Returns
        -------
        float
            Cross-validated score. Returns 0.0 if no features selected.
        """
        key = mask.astype(np.uint8).tobytes()
        if key in self._cache:
            return self._cache[key]

        selected = np.where(mask)[0]
        if len(selected) == 0:
            self._cache[key] = 0.0
            return 0.0

        cfg = self.cfg
        X_sub = self.X[:, selected]
        cv = (StratifiedKFold if cfg.task == "classification" else KFold)(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )

        scores: List[float] = []
        for train_idx, val_idx in cv.split(X_sub, self.y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_sub[train_idx])
            X_va = scaler.transform(X_sub[val_idx])
            y_tr, y_va = self.y[train_idx], self.y[val_idx]

            if cfg.task == "classification":
                model = RandomForestClassifier(
                    n_estimators=30, random_state=cfg.random_state, n_jobs=cfg.n_jobs
                )
                model.fit(X_tr, y_tr)
                if len(np.unique(y_va)) < 2:
                    scores.append(float(np.mean(model.predict(X_va) == y_va)))
                else:
                    scores.append(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]))
            else:
                model = RandomForestRegressor(
                    n_estimators=30, random_state=cfg.random_state, n_jobs=cfg.n_jobs
                )
                model.fit(X_tr, y_tr)
                scores.append(r2_score(y_va, model.predict(X_va)))

        raw_score = float(np.mean(scores))
        penalty = cfg.alpha * (len(selected) / self.X.shape[1])
        fitness = raw_score - penalty if not cfg.multi_objective else raw_score

        if len(self._cache) < FITNESS_CACHE_MAX_SIZE:
            self._cache[key] = fitness
        self._n_evaluations += 1
        return fitness

    @property
    def n_evaluations(self) -> int:
        return self._n_evaluations


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------
def _pareto_front(population: List[Individual]) -> List[Individual]:
    """Extract the non-dominated Pareto front from population."""
    front: List[Individual] = []
    for ind in population:
        dominated = any(other.dominates(ind) for other in population if other is not ind)
        if not dominated:
            front.append(ind)
    return front


def _population_diversity(population: List[Individual]) -> float:
    """Mean pairwise Hamming distance (normalised by n_features)."""
    masks = np.array([ind.mask for ind in population], dtype=float)
    n = len(masks)
    if n < 2:
        return 1.0
    n_feat = masks.shape[1]
    diffs = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diffs += np.sum(masks[i] != masks[j]) / n_feat
            count += 1
    return diffs / count


# ---------------------------------------------------------------------------
# GA
# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    """Standard binary GA with tournament selection, uniform crossover, bit-flip mutation."""

    def __init__(self, evaluator: FitnessEvaluator, cfg: EvoConfig, rng: np.random.Generator) -> None:
        self.ev = evaluator
        self.cfg = cfg
        self.rng = rng

    def _tournament(self, pop: List[Individual]) -> Individual:
        contestants = self.rng.choice(len(pop), size=self.cfg.ga_tournament_size, replace=False)  # type: ignore[arg-type]
        return max((pop[i] for i in contestants), key=lambda ind: ind.fitness)

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.cfg.crossover_prob:
            return p1.copy(), p2.copy()
        mask = self.rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        flip = self.rng.random(len(chromosome)) < self.cfg.mutation_prob
        child = chromosome.copy()
        child[flip] = 1 - child[flip]
        if child.sum() == 0:
            child[self.rng.integers(len(child))] = 1
        return child

    def evolve(self, population: List[Individual]) -> List[Individual]:
        """Produce next generation."""
        new_pop: List[Individual] = []
        while len(new_pop) < len(population):
            p1 = self._tournament(population)
            p2 = self._tournament(population)
            c1_mask, c2_mask = self._crossover(p1.mask, p2.mask)
            for child_mask in (self._mutate(c1_mask), self._mutate(c2_mask)):
                if len(new_pop) >= len(population):
                    break
                fitness = self.ev.evaluate(child_mask)
                new_pop.append(Individual(mask=child_mask, fitness=fitness))
        return new_pop


# ---------------------------------------------------------------------------
# Binary PSO
# ---------------------------------------------------------------------------
class BinaryPSO:
    """Binary PSO (S-shaped transfer function, velocity in continuous space)."""

    def __init__(self, evaluator: FitnessEvaluator, cfg: EvoConfig, rng: np.random.Generator) -> None:
        self.ev = evaluator
        self.cfg = cfg
        self.rng = rng

    def step(
        self,
        population: List[Individual],
        velocities: np.ndarray,
        personal_bests: List[Individual],
        global_best: Individual,
    ) -> Tuple[List[Individual], np.ndarray, List[Individual], Individual]:
        """Single BPSO iteration; returns updated (population, velocities, pbests, gbest)."""
        cfg = self.cfg
        n_feat = population[0].mask.shape[0]
        new_pop = []

        for i, (ind, vel) in enumerate(zip(population, velocities)):
            r1 = self.rng.random(n_feat)
            r2 = self.rng.random(n_feat)
            vel = (
                cfg.pso_inertia * vel
                + cfg.pso_c1 * r1 * (personal_bests[i].mask - ind.mask)
                + cfg.pso_c2 * r2 * (global_best.mask - ind.mask)
            )
            # S-shape transfer
            prob = 1.0 / (1.0 + np.exp(-vel))
            new_mask = (self.rng.random(n_feat) < prob).astype(int)
            if new_mask.sum() == 0:
                new_mask[self.rng.integers(n_feat)] = 1

            velocities[i] = vel
            fitness = self.ev.evaluate(new_mask)
            new_ind = Individual(mask=new_mask, fitness=fitness)
            new_pop.append(new_ind)

            if new_ind.fitness > personal_bests[i].fitness:
                personal_bests[i] = new_ind.clone()
            if new_ind.fitness > global_best.fitness:
                global_best = new_ind.clone()

        return new_pop, velocities, personal_bests, global_best


# ---------------------------------------------------------------------------
# Differential Evolution
# ---------------------------------------------------------------------------
class DifferentialEvolution:
    """DE/rand/1/bin adapted for binary feature masks."""

    def __init__(self, evaluator: FitnessEvaluator, cfg: EvoConfig, rng: np.random.Generator) -> None:
        self.ev = evaluator
        self.cfg = cfg
        self.rng = rng

    def evolve(self, population: List[Individual]) -> List[Individual]:
        """DE mutation + crossover + selection for one generation."""
        n = len(population)
        n_feat = population[0].mask.shape[0]
        new_pop = []

        for i, target in enumerate(population):
            indices = [j for j in range(n) if j != i]
            a, b, c = self.rng.choice(indices, size=3, replace=False)  # type: ignore[arg-type]

            mutant_cont = (
                population[a].mask.astype(float)
                + self.cfg.de_f * (population[b].mask.astype(float) - population[c].mask.astype(float))
            )
            prob = 1.0 / (1.0 + np.exp(-mutant_cont))
            trial_mask = (self.rng.random(n_feat) < prob).astype(int)

            # Crossover
            cross = self.rng.random(n_feat) < self.cfg.de_cr
            trial_mask = np.where(cross, trial_mask, target.mask)
            if trial_mask.sum() == 0:
                trial_mask[self.rng.integers(n_feat)] = 1

            trial_fitness = self.ev.evaluate(trial_mask)
            trial = Individual(mask=trial_mask, fitness=trial_fitness)

            new_pop.append(trial if trial.fitness >= target.fitness else target.clone())

        return new_pop


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------
class EvolutionaryFeatureSelector:
    """Unified evolutionary feature selector (GA / BPSO / DE).

    Parameters
    ----------
    config : EvoConfig

    Attributes
    ----------
    best_mask_ : np.ndarray
        Binary mask for the best single solution found.
    best_fitness_ : float
    pareto_front_ : list[Individual]
        Non-dominated solutions (only populated when multi_objective=True).
    history_ : list[float]
        Best fitness per generation.
    diversity_history_ : list[float]
        Population diversity per generation.
    """

    def __init__(self, config: EvoConfig) -> None:
        self.config = config
        self.best_mask_: Optional[np.ndarray] = None
        self.best_fitness_: float = -np.inf
        self.pareto_front_: List[Individual] = []
        self.history_: List[float] = []
        self.diversity_history_: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EvolutionaryFeatureSelector":
        """Run evolutionary optimisation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        self
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.random_state)
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n_feat = X.shape[1]

        evaluator = FitnessEvaluator(X, y, cfg)

        # Initialise population
        population = self._init_population(n_feat, cfg.population_size, rng, evaluator)

        # Algorithm-specific state
        velocities: Optional[np.ndarray] = None
        personal_bests: Optional[List[Individual]] = None
        bpso: Optional[BinaryPSO] = None

        if cfg.algorithm == "GA":
            operator = GeneticAlgorithm(evaluator, cfg, rng)
        elif cfg.algorithm == "BPSO":
            velocities = rng.standard_normal((cfg.population_size, n_feat)) * 0.1
            personal_bests = [ind.clone() for ind in population]
            bpso = BinaryPSO(evaluator, cfg, rng)
        elif cfg.algorithm == "DE":
            operator = DifferentialEvolution(evaluator, cfg, rng)
        else:
            raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

        global_best = max(population, key=lambda ind: ind.fitness)
        no_improve_count = 0

        t0 = time.perf_counter()
        for gen in range(cfg.n_generations):
            if cfg.algorithm == "GA":
                population = operator.evolve(population)  # type: ignore[union-attr]
            elif cfg.algorithm == "BPSO":
                population, velocities, personal_bests, global_best = bpso.step(  # type: ignore[misc]
                    population, velocities, personal_bests, global_best  # type: ignore[arg-type]
                )
            elif cfg.algorithm == "DE":
                population = operator.evolve(population)  # type: ignore[union-attr]

            gen_best = max(population, key=lambda ind: ind.fitness)
            if gen_best.fitness > global_best.fitness:
                global_best = gen_best.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            diversity = _population_diversity(population)
            self.history_.append(global_best.fitness)
            self.diversity_history_.append(diversity)

            if diversity < DIVERSITY_HAMMING_THRESHOLD:
                logger.debug("Gen %d: low diversity (%.4f). Consider restart.", gen, diversity)

            logger.debug(
                "Gen %d/%d | best_fitness=%.4f | diversity=%.4f | evals=%d",
                gen + 1, cfg.n_generations, global_best.fitness, diversity, evaluator.n_evaluations,
            )

            if no_improve_count >= cfg.convergence_patience:
                logger.info("Converged at generation %d (patience=%d).", gen, cfg.convergence_patience)
                break

        elapsed = time.perf_counter() - t0
        logger.info(
            "Evolution complete in %.2fs | algorithm=%s | best_fitness=%.4f | n_selected=%d",
            elapsed, cfg.algorithm, global_best.fitness, global_best.n_features,
        )

        self.best_mask_ = global_best.mask.astype(bool)
        self.best_fitness_ = global_best.fitness

        if cfg.multi_objective:
            self.pareto_front_ = _pareto_front(population)
            self.pareto_front_.sort(key=lambda ind: ind.n_features)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply best mask to X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples, n_selected)
        """
        if self.best_mask_ is None:
            raise RuntimeError("Call fit() first.")
        return np.asarray(X)[:, self.best_mask_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def pareto_summary(self) -> "pd.DataFrame":
        """Return Pareto front as a DataFrame (requires multi_objective=True)."""
        import pandas as pd
        if not self.config.multi_objective:
            raise RuntimeError("multi_objective=False; no Pareto front available.")
        rows = [
            {"n_features": ind.n_features, "fitness": ind.fitness,
             "mask": ind.mask.tolist()}
            for ind in self.pareto_front_
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _init_population(
        n_feat: int,
        pop_size: int,
        rng: np.random.Generator,
        evaluator: FitnessEvaluator,
    ) -> List[Individual]:
        pop = []
        for _ in range(pop_size):
            mask = (rng.random(n_feat) < 0.5).astype(int)
            if mask.sum() == 0:
                mask[rng.integers(n_feat)] = 1
            fitness = evaluator.evaluate(mask)
            pop.append(Individual(mask=mask, fitness=fitness))
        return pop


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=20, n_informative=5, random_state=0)

    for algo in ["GA", "BPSO", "DE"]:
        cfg = EvoConfig(
            algorithm=algo,
            population_size=20,
            n_generations=15,
            task="classification",
            cv_folds=3,
            multi_objective=False,
            alpha=0.02,
            convergence_patience=5,
        )
        evo = EvolutionaryFeatureSelector(cfg)
        evo.fit(X, y)
        selected = np.where(evo.best_mask_)[0]
        print(f"{algo}: fitness={evo.best_fitness_:.4f} | selected={selected.tolist()}")

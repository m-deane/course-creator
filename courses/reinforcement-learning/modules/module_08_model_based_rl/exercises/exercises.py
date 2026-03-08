"""
Module 08 - Model-Based Reinforcement Learning: Self-Check Exercises

Covers:
- Environment model: storing and retrieving transitions
- Dyna-Q planning step using a learned model
- UCB1 score computation for MCTS node selection

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Exercise 1: Simple Environment Model
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    Tabular environment model that stores and retrieves transitions.

    Problem
    -------
    A model-based agent maintains an internal model of the environment.
    For tabular problems, the simplest model maps (state, action) pairs
    to observed (next_state, reward) outcomes.

    This model stores the *most recent* observed (next_state, reward) for
    each (state, action) pair (a deterministic one-step model). A more
    complex model would maintain counts or distributions.

    Implement:
      - update(state, action, reward, next_state): store the transition.
      - sample(state, action): return (next_state, reward) for this pair.
      - sample_random_sa(): return a random (state, action) that has been
        observed at least once.
      - has_seen(state, action): True if (s,a) was observed.

    Hints
    -----
    - Store transitions in a dict: {(state, action): (next_state, reward)}.
    - random.choice on a list of keys for sample_random_sa.

    Examples
    --------
    >>> model = TransitionModel()
    >>> model.update(0, 1, reward=1.0, next_state=2)
    >>> model.sample(0, 1)
    (2, 1.0)
    """

    def __init__(self) -> None:
        # SOLUTION
        self._model: dict[tuple[int, int], tuple[int, float]] = {}

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """Store the most recent (next_state, reward) for (state, action)."""
        # SOLUTION
        self._model[(state, action)] = (next_state, reward)

    def sample(self, state: int, action: int) -> tuple[int, float]:
        """
        Retrieve the stored (next_state, reward) for the given (s, a).

        Raises
        ------
        KeyError
            If (state, action) has not been observed.
        """
        # SOLUTION
        return self._model[(state, action)]

    def has_seen(self, state: int, action: int) -> bool:
        """Return True if (state, action) has been observed at least once."""
        # SOLUTION
        return (state, action) in self._model

    def sample_random_sa(self, rng: np.random.Generator) -> tuple[int, int]:
        """
        Return a uniformly random (state, action) from observed pairs.

        Raises
        ------
        ValueError
            If the model is empty.
        """
        # SOLUTION
        if not self._model:
            raise ValueError("Model is empty — no transitions stored.")
        keys = list(self._model.keys())
        idx = rng.integers(0, len(keys))
        return keys[idx]


def test_exercise_1() -> None:
    model = TransitionModel()

    # Empty model: has_seen returns False
    assert not model.has_seen(0, 0), \
        "Empty model must return False for has_seen."

    # Update and retrieve
    model.update(state=0, action=1, reward=1.0, next_state=2)
    assert model.has_seen(0, 1), "After update, has_seen must return True."
    ns, r = model.sample(0, 1)
    assert ns == 2 and r == 1.0, \
        f"sample(0,1) must return (2, 1.0), got ({ns}, {r})."

    # Overwrite: most recent wins
    model.update(state=0, action=1, reward=5.0, next_state=3)
    ns2, r2 = model.sample(0, 1)
    assert ns2 == 3 and r2 == 5.0, \
        f"Model must store most recent transition; got ({ns2}, {r2})."

    # sample_random_sa returns a seen pair
    model.update(0, 0, 0.0, 1)
    rng = np.random.default_rng(42)
    sa = model.sample_random_sa(rng)
    assert model.has_seen(*sa), \
        f"sample_random_sa must return an observed (s,a), got {sa}."

    # KeyError on unseen (s, a)
    raised = False
    try:
        model.sample(99, 99)
    except KeyError:
        raised = True
    assert raised, "sample on unseen (s,a) must raise KeyError."

    # ValueError on empty model
    raised2 = False
    try:
        TransitionModel().sample_random_sa(np.random.default_rng(0))
    except ValueError:
        raised2 = True
    assert raised2, "sample_random_sa on empty model must raise ValueError."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Dyna-Q Planning Step
# ---------------------------------------------------------------------------

def dyna_q_planning_step(
    Q: np.ndarray,
    model: TransitionModel,
    alpha: float,
    gamma: float,
    n_planning_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Perform n_planning_steps of Dyna-Q model-based updates.

    Problem
    -------
    Dyna-Q interleaves real experience with simulated (planning) steps.
    Each planning step:
      1. Sample a random previously-seen (s, a) from the model.
      2. Retrieve the simulated (s', r) from the model.
      3. Apply a Q-learning update:
             Q[s, a] += alpha * (r + gamma * max_a' Q[s'] - Q[s, a])

    Repeat n_planning_steps times. The Q-table is modified in-place.

    Hints
    -----
    - Use model.sample_random_sa(rng) and model.sample(s, a).
    - The planning update is identical to online Q-learning except
      (s, r, s') come from the model, not the real environment.

    Parameters
    ----------
    Q : np.ndarray, shape (S, A)
        Current Q-table (modified in-place and returned).
    model : TransitionModel
        Learned environment model.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    n_planning_steps : int
        Number of simulated updates to perform.
    rng : np.random.Generator
        Random generator for reproducible sampling.

    Returns
    -------
    np.ndarray, shape (S, A)
        Updated Q-table.
    """
    # SOLUTION
    for _ in range(n_planning_steps):
        s, a = model.sample_random_sa(rng)
        next_s, reward = model.sample(s, a)
        td_target = reward + gamma * np.max(Q[next_s])
        Q[s, a] += alpha * (td_target - Q[s, a])
    return Q


def test_exercise_2() -> None:
    # Simple model: (0,0) -> (1, 1.0); (1,0) -> (1, 0.0) absorbing
    model = TransitionModel()
    model.update(0, 0, reward=1.0, next_state=1)
    model.update(1, 0, reward=0.0, next_state=1)

    Q = np.zeros((2, 1))
    rng = np.random.default_rng(42)

    # After many planning steps, Q[0,0] should increase toward optimal
    Q = dyna_q_planning_step(Q, model, alpha=0.5, gamma=0.9,
                               n_planning_steps=100, rng=rng)

    # Q[1,0] absorbing: Q[1,0] stays close to 0 (no reward in self-loop)
    assert Q[1, 0] >= -1e-6, \
        f"Q[1,0] (absorbing, R=0) must be near 0, got {Q[1,0]}."
    # Q[0,0] must have grown from the reward signal
    assert Q[0, 0] > 0.5, \
        f"Q[0,0] must grow after 100 planning steps toward ~1, got {Q[0,0]}."

    # Shape preservation
    assert Q.shape == (2, 1), f"Q shape must remain (2,1), got {Q.shape}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: UCB1 Score for MCTS Node Selection
# ---------------------------------------------------------------------------

def ucb1_scores(
    q_values: np.ndarray,
    visit_counts: np.ndarray,
    parent_visits: int,
    c: float = np.sqrt(2),
) -> np.ndarray:
    """
    Compute UCB1 scores for selecting child nodes in MCTS.

    Problem
    -------
    Upper Confidence Bound (UCB1) balances exploitation and exploration
    during tree search:

        UCB1(child_i) = Q(child_i) + c * sqrt(ln(N) / n_i)

    where:
      - Q(child_i) is the average reward (exploitation term).
      - N = parent_visits is the total number of parent visits.
      - n_i = visit_counts[i] is the visit count of child i.
      - c controls exploration (default sqrt(2) for theoretical UCB1).

    Unvisited children (n_i = 0) have UCB1 = +infinity to ensure they
    are visited before any visited child.

    Hints
    -----
    - Use np.where to assign inf to unvisited children.
    - Use np.log(parent_visits) for the log term.
    - Add small epsilon inside sqrt to avoid numerical issues (optional).

    Parameters
    ----------
    q_values : np.ndarray, shape (K,)
        Mean Q-values for each child node.
    visit_counts : np.ndarray, shape (K,)
        Number of times each child has been visited.
    parent_visits : int
        Total visits to the parent node.
    c : float
        Exploration constant (default sqrt(2)).

    Returns
    -------
    np.ndarray, shape (K,)
        UCB1 score for each child. Unvisited children have score +inf.

    Examples
    --------
    >>> ucb1_scores(np.array([0.5, 0.0]), np.array([4, 0]), parent_visits=4)
    array([0.5 + sqrt(2)*sqrt(ln4/4), inf])
    """
    # SOLUTION
    # Suppress divide-by-zero: unvisited nodes (n=0) are masked to inf anyway.
    with np.errstate(divide="ignore", invalid="ignore"):
        exploration = c * np.sqrt(np.log(parent_visits) / visit_counts)
    scores = np.where(visit_counts == 0, np.inf, q_values + exploration)
    return scores


def test_exercise_3() -> None:
    # Unvisited child must have score inf
    scores = ucb1_scores(
        np.array([0.5, 0.0]),
        np.array([4, 0]),
        parent_visits=4,
    )
    assert scores.shape == (2,), f"Shape must be (2,), got {scores.shape}."
    assert np.isinf(scores[1]), \
        f"Unvisited child must have score=inf, got {scores[1]}."

    # UCB1 of visited child: Q + c * sqrt(ln(N)/n)
    expected_0 = 0.5 + np.sqrt(2) * np.sqrt(np.log(4) / 4)
    assert np.isclose(scores[0], expected_0), \
        f"UCB1 score[0] must be {expected_0:.4f}, got {scores[0]:.4f}."

    # All visited with equal Q: prefer less-visited child (higher bonus)
    scores2 = ucb1_scores(
        np.array([0.0, 0.0, 0.0]),
        np.array([10, 5, 1]),
        parent_visits=16,
    )
    assert np.argmax(scores2) == 2, \
        f"Least-visited child must have highest UCB1, got argmax={np.argmax(scores2)}."

    # Higher c -> more exploration (higher scores for low-visit nodes)
    s_low_c = ucb1_scores(np.array([0.0]), np.array([2]), parent_visits=10, c=0.1)
    s_high_c = ucb1_scores(np.array([0.0]), np.array([2]), parent_visits=10, c=5.0)
    assert s_high_c[0] > s_low_c[0], \
        "Higher c must produce higher UCB1 scores."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: Environment Model (Store & Retrieve)", test_exercise_1),
        ("Exercise 2: Dyna-Q Planning Step", test_exercise_2),
        ("Exercise 3: UCB1 Scores for MCTS", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

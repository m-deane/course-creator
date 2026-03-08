"""
Module 02 - Monte Carlo Methods: Self-Check Exercises

Covers:
- First-visit MC value estimation from episodes
- Importance sampling ratio for off-policy evaluation
- Incremental (running average) MC update

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: First-Visit MC Estimate
# ---------------------------------------------------------------------------

def first_visit_mc(
    episodes: list[list[tuple[int, float]]],
    num_states: int,
    gamma: float,
) -> np.ndarray:
    """
    Estimate V(s) using first-visit Monte Carlo averaging.

    Problem
    -------
    Each episode is a list of (state, reward) tuples representing a
    trajectory: [(s_0, r_1), (s_1, r_2), ..., (s_{T-1}, r_T)].
    The reward r_{t+1} is received after taking the step from s_t.

    For each episode, compute the discounted return G_t from each
    time step t. For every state, only the **first** visit within each
    episode contributes to the running average.

    Steps:
      1. For each episode, scan backwards to compute G_t at each step.
      2. Identify the first visit of each state in the episode.
      3. Average the returns across all first visits over all episodes.

    Hints
    -----
    - Use a set "seen_states" reset per episode to track first visits.
    - If a state is never visited across all episodes, leave V[s] = 0.

    Parameters
    ----------
    episodes : list of list of (int, float)
        Collection of episodes. Each inner list is
        [(s_0, r_1), (s_1, r_2), ..., (s_{T-1}, r_T)].
    num_states : int
        Total number of states (for output array shape).
    gamma : float
        Discount factor in [0, 1].

    Returns
    -------
    np.ndarray, shape (num_states,)
        Estimated V[s] for each state. States never visited have V[s] = 0.

    Examples
    --------
    >>> episodes = [[(0, 1.0), (1, 1.0)]]  # two steps, both reward 1
    >>> first_visit_mc(episodes, num_states=2, gamma=0.9)
    array([1.9, 1.0])  # V[0]=1+0.9*1=1.9, V[1]=1.0
    """
    # SOLUTION
    returns: dict[int, list[float]] = {s: [] for s in range(num_states)}

    for episode in episodes:
        T = len(episode)
        # Compute returns backward
        G = 0.0
        Gs = []
        for t in reversed(range(T)):
            _, r = episode[t]
            G = r + gamma * G
            Gs.append(G)
        Gs = list(reversed(Gs))  # Gs[t] = discounted return from step t

        seen: set[int] = set()
        for t, (s, _) in enumerate(episode):
            if s not in seen:
                returns[s].append(Gs[t])
                seen.add(s)

    V = np.array([
        np.mean(returns[s]) if returns[s] else 0.0
        for s in range(num_states)
    ])
    return V


def test_exercise_1() -> None:
    # Single episode: state 0 -> state 1, each with reward 1
    episodes = [[(0, 1.0), (1, 1.0)]]
    V = first_visit_mc(episodes, num_states=2, gamma=0.9)
    assert V.shape == (2,), f"Shape must be (2,), got {V.shape}."
    assert np.isclose(V[0], 1.9), f"V[0] must be 1.9, got {V[0]}."
    assert np.isclose(V[1], 1.0), f"V[1] must be 1.0, got {V[1]}."

    # gamma=0: each V[s] = immediate reward after first visit
    V_g0 = first_visit_mc(episodes, num_states=2, gamma=0.0)
    assert np.isclose(V_g0[0], 1.0), \
        f"gamma=0: V[0] must equal first reward r_1=1.0, got {V_g0[0]}."

    # Multiple episodes: first-visit only counts first occurrence
    # Episode: s0, s0, s1  ->  first visit s0: G=1+0.9*0+0.9^2*1=1.81, s1: G=1
    episode2 = [(0, 1.0), (0, 0.0), (1, 1.0)]
    V2 = first_visit_mc([episode2], num_states=2, gamma=0.9)
    # G at t=0 (first visit s0): 1 + 0.9*0 + 0.9^2*1 = 1 + 0.81 = 1.81
    assert np.isclose(V2[0], 1.81), \
        f"First-visit s0 return must be 1.81, got {V2[0]}."

    # Unvisited state stays 0
    assert V2[1] == 1.0  # s1 visited once at t=2

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Importance Sampling Ratio
# ---------------------------------------------------------------------------

def importance_sampling_ratio(
    trajectory: list[tuple[int, int]],
    target_policy: np.ndarray,
    behavior_policy: np.ndarray,
) -> float:
    """
    Compute the ordinary importance sampling ratio rho for a trajectory.

    Problem
    -------
    When evaluating a target policy pi using data collected under a
    behavior policy b, each trajectory must be re-weighted by:

        rho = product_{t=0}^{T-1} [ pi(a_t | s_t) / b(a_t | s_t) ]

    If b(a_t | s_t) = 0 for any step, the ratio is undefined; return 0.0
    in that case (the trajectory cannot support the target policy).

    Hints
    -----
    - Multiply ratios step by step; break early if denominator is zero.
    - Use np.prod on an array of per-step ratios.

    Parameters
    ----------
    trajectory : list of (int, int)
        Sequence of (state, action) pairs.
    target_policy : np.ndarray, shape (S, A)
        Target policy pi(a | s).
    behavior_policy : np.ndarray, shape (S, A)
        Behavior policy b(a | s).

    Returns
    -------
    float
        Importance sampling ratio rho.

    Examples
    --------
    >>> target  = np.array([[0.0, 1.0], [1.0, 0.0]])  # deterministic
    >>> behavior = np.array([[0.5, 0.5], [0.5, 0.5]]) # uniform
    >>> importance_sampling_ratio([(0, 1), (1, 0)], target, behavior)
    4.0   # (1/0.5) * (1/0.5)
    """
    # SOLUTION
    rho = 1.0
    for s, a in trajectory:
        b_sa = behavior_policy[s, a]
        if b_sa == 0.0:
            return 0.0
        rho *= target_policy[s, a] / b_sa
    return rho


def test_exercise_2() -> None:
    target = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    behavior = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float)

    # Trajectory consistent with target: rho = (1/0.5)*(1/0.5) = 4.0
    rho = importance_sampling_ratio([(0, 1), (1, 0)], target, behavior)
    assert np.isclose(rho, 4.0), f"rho must be 4.0, got {rho}."

    # Trajectory inconsistent with target: pi(a=0|s=0)=0 -> rho=0
    rho_zero = importance_sampling_ratio([(0, 0)], target, behavior)
    assert rho_zero == 0.0, \
        f"Trajectory impossible under target policy must give rho=0.0, got {rho_zero}."

    # Same target and behavior: rho = 1 for any trajectory
    uniform = np.full((2, 2), 0.5)
    rho_one = importance_sampling_ratio([(0, 0), (1, 1)], uniform, uniform)
    assert np.isclose(rho_one, 1.0), \
        f"Identical policies must give rho=1.0, got {rho_one}."

    # Empty trajectory: rho = 1 (empty product)
    rho_empty = importance_sampling_ratio([], target, behavior)
    assert rho_empty == 1.0, \
        f"Empty trajectory must give rho=1.0 (empty product), got {rho_empty}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Incremental MC Update (Running Average)
# ---------------------------------------------------------------------------

def incremental_mc_update(
    V: np.ndarray,
    counts: np.ndarray,
    state: int,
    G: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the running average estimate for a single state.

    Problem
    -------
    Instead of storing all returns and averaging at the end, maintain a
    running count N[s] and update V[s] incrementally after each new return:

        N[s] <- N[s] + 1
        V[s] <- V[s] + (G - V[s]) / N[s]

    This is mathematically equivalent to the batch mean but uses O(S)
    memory regardless of the number of episodes.

    Hints
    -----
    - Increment counts[state] before updating V[state].
    - The update rule (V + delta/N) is the standard running mean formula.

    Parameters
    ----------
    V : np.ndarray, shape (S,)
        Current value estimates (modified in-place and returned).
    counts : np.ndarray, shape (S,)
        Visit counts per state (modified in-place and returned).
    state : int
        Index of the state being updated.
    G : float
        Observed discounted return for this visit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (V, counts) arrays.

    Examples
    --------
    >>> V = np.zeros(3); counts = np.zeros(3)
    >>> V, counts = incremental_mc_update(V, counts, state=1, G=4.0)
    >>> V[1]      # (0 + 4.0) / 1 = 4.0
    4.0
    >>> V, counts = incremental_mc_update(V, counts, state=1, G=2.0)
    >>> V[1]      # (4.0 + (2.0 - 4.0) / 2) = 3.0
    3.0
    """
    # SOLUTION
    counts[state] += 1
    V[state] += (G - V[state]) / counts[state]
    return V, counts


def test_exercise_3() -> None:
    V = np.zeros(3)
    counts = np.zeros(3, dtype=float)

    # First update to state 1: V[1] = 4.0
    V, counts = incremental_mc_update(V, counts, state=1, G=4.0)
    assert counts[1] == 1, f"counts[1] must be 1 after first update, got {counts[1]}."
    assert np.isclose(V[1], 4.0), f"V[1] must be 4.0, got {V[1]}."

    # Second update to state 1: V[1] = (4+2)/2 = 3.0
    V, counts = incremental_mc_update(V, counts, state=1, G=2.0)
    assert counts[1] == 2, f"counts[1] must be 2 after second update."
    assert np.isclose(V[1], 3.0), f"V[1] must be 3.0 (average of 4 and 2), got {V[1]}."

    # Other states unaffected
    assert V[0] == 0.0 and V[2] == 0.0, \
        "Unupdated states must remain 0."

    # Converges to true mean over many updates
    V2 = np.zeros(1)
    c2 = np.zeros(1, dtype=float)
    returns = [float(x) for x in range(1, 101)]  # 1..100, mean=50.5
    for g in returns:
        V2, c2 = incremental_mc_update(V2, c2, state=0, G=g)
    assert np.isclose(V2[0], 50.5), \
        f"Running average over 1..100 must be 50.5, got {V2[0]}."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: First-Visit MC Estimate", test_exercise_1),
        ("Exercise 2: Importance Sampling Ratio", test_exercise_2),
        ("Exercise 3: Incremental MC Update", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

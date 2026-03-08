"""
Module 00 - Foundations of Reinforcement Learning: Self-Check Exercises

Covers:
- Discounted return computation
- Markov decision process (MDP) transition matrix validation
- Epsilon-greedy action selection
- State-value function computation via direct Bellman equation solve

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: Discounted Return
# ---------------------------------------------------------------------------

def compute_discounted_return(rewards: list[float], gamma: float) -> float:
    """
    Compute the discounted return G_t = sum_{k=0}^{T} gamma^k * r_{t+k}.

    Problem
    -------
    Given a sequence of rewards [r_0, r_1, ..., r_{T-1}] and a discount
    factor gamma in [0, 1], return the total discounted return starting
    from the first reward.

    Hints
    -----
    - Iterate rewards in reverse order; this avoids computing gamma^k
      explicitly and runs in O(T) time.
    - G = r_T, then for each step backwards: G = r_t + gamma * G

    Parameters
    ----------
    rewards : list of float
        Sequence of rewards [r_0, r_1, ..., r_{T-1}].
    gamma : float
        Discount factor in [0, 1].

    Returns
    -------
    float
        Discounted return G_0.

    Examples
    --------
    >>> compute_discounted_return([1.0, 1.0, 1.0], 0.9)
    2.71  # 1 + 0.9 + 0.81
    >>> compute_discounted_return([1.0], 0.5)
    1.0
    """
    # SOLUTION
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


def test_exercise_1() -> None:
    # Single reward
    assert compute_discounted_return([1.0], 0.9) == 1.0, \
        "Single reward with any gamma must equal the reward itself."

    # gamma=0: only first reward counts
    result = compute_discounted_return([2.0, 5.0, 10.0], 0.0)
    assert result == 2.0, \
        f"gamma=0 must return only r_0=2.0, got {result}."

    # gamma=1: undiscounted sum
    result = compute_discounted_return([1.0, 2.0, 3.0], 1.0)
    assert result == 6.0, \
        f"gamma=1 must return sum of rewards=6.0, got {result}."

    # Standard case: [1, 1, 1], gamma=0.9 -> 1 + 0.9 + 0.81 = 2.71
    result = compute_discounted_return([1.0, 1.0, 1.0], 0.9)
    assert abs(result - 2.71) < 1e-9, \
        f"Expected 2.71, got {result}."

    # Empty rewards list
    result = compute_discounted_return([], 0.9)
    assert result == 0.0, \
        f"Empty rewards must return 0.0, got {result}."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Transition Matrix Validation
# ---------------------------------------------------------------------------

def is_valid_transition_matrix(P: np.ndarray) -> bool:
    """
    Check whether P is a valid stochastic (transition) matrix.

    Problem
    -------
    A valid transition matrix P of shape (S, S) must satisfy:
      1. All entries are non-negative: P[s, s'] >= 0 for all s, s'.
      2. Each row sums to exactly 1: sum_s' P[s, s'] == 1 for all s.

    Hints
    -----
    - Use np.allclose for floating-point row-sum comparison.
    - A single np.all(...) call on element-wise conditions is idiomatic.

    Parameters
    ----------
    P : np.ndarray, shape (S, S)
        Candidate transition matrix.

    Returns
    -------
    bool
        True if P is a valid stochastic matrix, False otherwise.

    Examples
    --------
    >>> is_valid_transition_matrix(np.array([[0.3, 0.7], [1.0, 0.0]]))
    True
    >>> is_valid_transition_matrix(np.array([[0.3, 0.8], [1.0, 0.0]]))
    False  # first row sums to 1.1
    """
    # SOLUTION
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    non_negative = np.all(P >= 0)
    rows_sum_to_one = np.allclose(P.sum(axis=1), np.ones(P.shape[0]))
    return bool(non_negative and rows_sum_to_one)


def test_exercise_2() -> None:
    # Valid 2x2 matrix
    P_valid = np.array([[0.3, 0.7], [1.0, 0.0]])
    assert is_valid_transition_matrix(P_valid), \
        "Valid 2x2 stochastic matrix must return True."

    # Row does not sum to 1
    P_bad_sum = np.array([[0.3, 0.8], [1.0, 0.0]])
    assert not is_valid_transition_matrix(P_bad_sum), \
        "Matrix with row summing to 1.1 must return False."

    # Negative entry
    P_negative = np.array([[-0.1, 1.1], [0.5, 0.5]])
    assert not is_valid_transition_matrix(P_negative), \
        "Matrix with negative entry must return False."

    # Identity matrix is valid
    P_identity = np.eye(4)
    assert is_valid_transition_matrix(P_identity), \
        "4x4 identity matrix must return True."

    # Uniform 3x3
    P_uniform = np.full((3, 3), 1.0 / 3.0)
    assert is_valid_transition_matrix(P_uniform), \
        "Uniform 3x3 stochastic matrix must return True."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Epsilon-Greedy Action Selection
# ---------------------------------------------------------------------------

def epsilon_greedy(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """
    Select an action using the epsilon-greedy policy.

    Problem
    -------
    Given an array of Q-values q_values[a] for each action a, return:
      - A uniformly random action with probability epsilon.
      - The greedy (argmax) action with probability (1 - epsilon).

    When multiple actions share the maximum Q-value, any one of them may
    be returned as the greedy choice (np.argmax returns the first by default,
    which is acceptable).

    Hints
    -----
    - Draw a single uniform sample from rng to decide explore vs exploit.
    - rng.integers(low, high) draws a random integer in [low, high).

    Parameters
    ----------
    q_values : np.ndarray, shape (A,)
        Q-value estimate for each of the A actions.
    epsilon : float
        Exploration probability in [0, 1].
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    int
        Selected action index.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> epsilon_greedy(np.array([1.0, 5.0, 2.0]), 0.0, rng)
    1  # always greedy when epsilon=0
    """
    # SOLUTION
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))
    return int(np.argmax(q_values))


def test_exercise_3() -> None:
    q = np.array([1.0, 5.0, 2.0])

    # epsilon=0 must always return greedy action (index 1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = epsilon_greedy(q, 0.0, rng)
        assert action == 1, \
            f"epsilon=0 must always select greedy action 1, got {action}."

    # epsilon=1 must return valid action indices only
    rng = np.random.default_rng(0)
    actions = [epsilon_greedy(q, 1.0, rng) for _ in range(200)]
    assert all(0 <= a < len(q) for a in actions), \
        "epsilon=1 must return valid action indices."

    # epsilon=1: over many samples each action should appear
    unique_actions = set(actions)
    assert len(unique_actions) == len(q), \
        f"epsilon=1 should explore all actions over 200 trials, got {unique_actions}."

    # Returned value is int
    rng = np.random.default_rng(7)
    result = epsilon_greedy(q, 0.5, rng)
    assert isinstance(result, int), \
        f"Return type must be int, got {type(result)}."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Exercise 4: State-Value Function via Direct Bellman Solve
# ---------------------------------------------------------------------------

def compute_state_values(
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute the exact state-value function V^pi by solving the Bellman
    equations as a linear system.

    Problem
    -------
    For a policy pi, the Bellman expectation equation is:

        V^pi = R^pi + gamma * P^pi @ V^pi

    where:
      - P^pi[s, s'] = sum_a pi[s, a] * P[s, a, s']  (policy-weighted transitions)
      - R^pi[s]     = sum_a pi[s, a] * R[s, a]       (policy-weighted rewards)

    Rearranging:
        (I - gamma * P^pi) @ V^pi = R^pi

    Solve this linear system using np.linalg.solve.

    Hints
    -----
    - Use np.einsum or matrix multiplication to compute P^pi and R^pi.
    - P^pi has shape (S, S); R^pi has shape (S,).

    Parameters
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition probability tensor. P[s, a, s'] = Pr(s' | s, a).
    R : np.ndarray, shape (S, A)
        Expected reward matrix. R[s, a] = E[r | s, a].
    policy : np.ndarray, shape (S, A)
        Stochastic policy. policy[s, a] = pi(a | s); rows sum to 1.
    gamma : float
        Discount factor in [0, 1).

    Returns
    -------
    np.ndarray, shape (S,)
        State-value function V^pi[s] for each state s.

    Examples
    --------
    A 2-state MDP where always going right (action 0) gives reward 1
    and deterministically transitions to the other state in a loop.
    """
    # SOLUTION
    S = P.shape[0]
    # P^pi[s, s'] = sum_a pi[s,a] * P[s,a,s']
    # P has shape (S, A, S); use axis notation: "sa,sat->st" (t = next state)
    P_pi = np.einsum("sa,sat->st", policy, P)     # (S, S)
    # R^pi[s] = sum_a pi[s,a] * R[s,a]
    R_pi = np.einsum("sa,sa->s", policy, R)       # (S,)
    A_mat = np.eye(S) - gamma * P_pi
    V = np.linalg.solve(A_mat, R_pi)
    return V


def test_exercise_4() -> None:
    # 2-state MDP, 1 action, deterministic cycle s0->s1->s0
    # R[s,a] = 1 for all; gamma=0.5
    # V[s] = 1 + 0.5 * V[s'] (by symmetry V[0] = V[1] = 2.0)
    S, A = 2, 1
    P = np.zeros((S, A, S))
    P[0, 0, 1] = 1.0
    P[1, 0, 0] = 1.0
    R = np.ones((S, A))
    policy = np.ones((S, A))  # single action; trivially sums to 1
    V = compute_state_values(P, R, policy, gamma=0.5)
    assert V.shape == (S,), f"V must have shape ({S},), got {V.shape}."
    assert np.allclose(V, 2.0), \
        f"Expected V=[2.0, 2.0] for symmetric 2-state MDP, got {V}."

    # Absorbing state: R=0 everywhere, gamma=0.9 -> V=0 everywhere
    P2 = np.zeros((3, 2, 3))
    for s in range(3):
        P2[s, :, s] = 1.0  # all transitions self-loop (absorbing-like)
    R2 = np.zeros((3, 2))
    policy2 = np.full((3, 2), 0.5)
    V2 = compute_state_values(P2, R2, policy2, gamma=0.9)
    assert np.allclose(V2, 0.0), \
        f"Zero-reward MDP must have V=0, got {V2}."

    print("Exercise 4 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: Discounted Return", test_exercise_1),
        ("Exercise 2: Transition Matrix Validation", test_exercise_2),
        ("Exercise 3: Epsilon-Greedy Action Selection", test_exercise_3),
        ("Exercise 4: State-Value Function (Bellman Solve)", test_exercise_4),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

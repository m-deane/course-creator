"""
Module 01 - Dynamic Programming: Self-Check Exercises

Covers:
- Iterative policy evaluation (one Bellman backup sweep)
- Greedy policy improvement from a value function
- Value iteration for tabular MDPs
- Convergence comparison: policy iteration vs value iteration

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: One Sweep of Iterative Policy Evaluation
# ---------------------------------------------------------------------------

def policy_evaluation_sweep(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    policy: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Perform one synchronous sweep of the Bellman expectation backup.

    Problem
    -------
    Given current value estimates V[s], compute new estimates V_new[s]
    using the Bellman expectation equation for policy pi:

        V_new[s] = sum_a pi[s,a] * sum_s' P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])

    This is a single in-place sweep over all states. The update is
    synchronous (uses old V for all backups, writes to a new array).

    Hints
    -----
    - Iterate over states s and actions a explicitly, or use np.einsum.
    - R here has shape (S, A, S) — reward depends on (s, a, s').

    Parameters
    ----------
    V : np.ndarray, shape (S,)
        Current value estimates (not modified in-place).
    P : np.ndarray, shape (S, A, S)
        Transition tensor. P[s, a, s'] = Pr(s' | s, a).
    R : np.ndarray, shape (S, A, S)
        Reward tensor. R[s, a, s'] = expected reward for transition.
    policy : np.ndarray, shape (S, A)
        Stochastic policy pi(a | s); rows sum to 1.
    gamma : float
        Discount factor.

    Returns
    -------
    np.ndarray, shape (S,)
        Updated value estimates after one sweep.
    """
    # SOLUTION
    # Q[s, a] = sum_t P[s,a,t] * (R[s,a,t] + gamma * V[t])  (t = next state)
    immediate = np.einsum("sat,t->sa", P, gamma * V)           # (S, A)
    reward_term = np.einsum("sat,sat->sa", P, R)               # (S, A)
    Q = reward_term + immediate                                  # (S, A)
    V_new = np.einsum("sa,sa->s", policy, Q)                   # (S,)
    return V_new


def test_exercise_1() -> None:
    # 2 states, 1 action, deterministic: s0->s1->s0, R=1 everywhere
    S, A = 2, 1
    P = np.zeros((S, A, S))
    P[0, 0, 1] = 1.0
    P[1, 0, 0] = 1.0
    R = np.ones((S, A, S))
    policy = np.ones((S, A))
    V = np.zeros(S)
    gamma = 0.9

    V1 = policy_evaluation_sweep(V, P, R, policy, gamma)
    assert V1.shape == (S,), f"Shape must be ({S},), got {V1.shape}."
    # After one sweep from V=0: V_new[s] = 1 * P * (R + gamma*0) = 1.0
    assert np.allclose(V1, 1.0), \
        f"First sweep from V=0 must give V=[1.0,1.0], got {V1}."

    # Second sweep: V_new[s] = 1 + gamma * 1 = 1.9
    V2 = policy_evaluation_sweep(V1, P, R, policy, gamma)
    assert np.allclose(V2, 1.0 + gamma), \
        f"Second sweep must give V={1.0+gamma}, got {V2}."

    # Returns new array, does not modify V in place
    V_orig = np.zeros(S)
    _ = policy_evaluation_sweep(V_orig, P, R, policy, gamma)
    assert np.all(V_orig == 0.0), \
        "policy_evaluation_sweep must not modify the input V array."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Greedy Policy Improvement
# ---------------------------------------------------------------------------

def greedy_policy_improvement(
    V: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Derive a deterministic greedy policy from a value function.

    Problem
    -------
    For each state s, select the action that maximises the action-value:

        pi_new[s] = argmax_a sum_s' P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])

    Return a one-hot policy matrix where policy[s, a] = 1 if a = pi_new[s],
    else 0.

    Hints
    -----
    - Compute the full Q-table first, then take argmax over actions.
    - np.argmax on axis=1 gives the greedy action for each state.

    Parameters
    ----------
    V : np.ndarray, shape (S,)
        State-value estimates.
    P : np.ndarray, shape (S, A, S)
        Transition tensor.
    R : np.ndarray, shape (S, A, S)
        Reward tensor.
    gamma : float
        Discount factor.

    Returns
    -------
    np.ndarray, shape (S, A)
        Deterministic one-hot policy. Rows sum to 1, entries are 0 or 1.
    """
    # SOLUTION
    immediate = np.einsum("sat,t->sa", P, gamma * V)
    reward_term = np.einsum("sat,sat->sa", P, R)
    Q = reward_term + immediate                   # (S, A)
    greedy_actions = np.argmax(Q, axis=1)         # (S,)
    S, A = Q.shape
    policy = np.zeros((S, A))
    policy[np.arange(S), greedy_actions] = 1.0
    return policy


def test_exercise_2() -> None:
    # 2 states, 2 actions
    # Action 0: self-loop, R=0; Action 1: go to other state, R=1
    S, A = 2, 2
    P = np.zeros((S, A, S))
    # action 0: self-loop
    P[0, 0, 0] = 1.0
    P[1, 0, 1] = 1.0
    # action 1: cross to other state
    P[0, 1, 1] = 1.0
    P[1, 1, 0] = 1.0
    R = np.zeros((S, A, S))
    R[:, 1, :] = 1.0   # action 1 always gives reward 1

    V = np.zeros(S)
    policy = greedy_policy_improvement(V, P, R, gamma=0.9)
    assert policy.shape == (S, A), f"Policy shape must be ({S},{A})."
    assert np.allclose(policy.sum(axis=1), 1.0), \
        "Each row of policy must sum to 1."
    # Greedy action must be 1 for both states (higher reward)
    assert np.all(np.argmax(policy, axis=1) == 1), \
        f"Greedy action must be 1 (higher reward) for both states, got {np.argmax(policy, axis=1)}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Value Iteration
# ---------------------------------------------------------------------------

def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    theta: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, int]:
    """
    Solve an MDP with value iteration to find the optimal value function.

    Problem
    -------
    Value iteration repeatedly applies the Bellman optimality backup:

        V_new[s] = max_a sum_s' P[s,a,s'] * (R[s,a,s'] + gamma * V[s'])

    until the maximum change across all states is below theta.

    Hints
    -----
    - Compute the full Q-table; take max over actions.
    - Track delta = max |V_new - V| per sweep; stop when delta < theta.

    Parameters
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition tensor.
    R : np.ndarray, shape (S, A, S)
        Reward tensor.
    gamma : float
        Discount factor in [0, 1).
    theta : float
        Convergence threshold.
    max_iter : int
        Maximum number of sweeps.

    Returns
    -------
    tuple[np.ndarray, int]
        (V_star, n_iterations) where V_star is shape (S,) and
        n_iterations is the number of sweeps until convergence.
    """
    # SOLUTION
    S = P.shape[0]
    V = np.zeros(S)
    for i in range(max_iter):
        immediate = np.einsum("sat,t->sa", P, gamma * V)
        reward_term = np.einsum("sat,sat->sa", P, R)
        Q = reward_term + immediate
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < theta:
            return V, i + 1
    return V, max_iter


def test_exercise_3() -> None:
    # 3-state chain: s0->s1->s2 (absorbing). R=1 on s1->s2, 0 elsewhere.
    # Only one meaningful action (action 0 moves forward).
    S, A = 3, 1
    P = np.zeros((S, A, S))
    P[0, 0, 1] = 1.0
    P[1, 0, 2] = 1.0
    P[2, 0, 2] = 1.0   # absorbing
    R = np.zeros((S, A, S))
    R[1, 0, 2] = 1.0   # reward on s1->s2
    gamma = 0.9

    V_star, n_iter = value_iteration(P, R, gamma)
    assert V_star.shape == (S,), f"V_star shape must be ({S},)."
    assert isinstance(n_iter, int) and n_iter > 0, \
        f"n_iter must be a positive int, got {n_iter}."
    # V[s2]=0 (absorbing, R=0 self-loop), V[s1]=1+0*0=1, V[s0]=0+gamma*1=0.9
    assert np.isclose(V_star[2], 0.0, atol=1e-6), \
        f"V[s2] must be 0 (absorbing), got {V_star[2]}."
    assert np.isclose(V_star[1], 1.0, atol=1e-6), \
        f"V[s1] must be 1.0, got {V_star[1]}."
    assert np.isclose(V_star[0], gamma, atol=1e-6), \
        f"V[s0] must be gamma={gamma}, got {V_star[0]}."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Exercise 4: Policy Iteration vs Value Iteration Convergence
# ---------------------------------------------------------------------------

def policy_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    eval_theta: float = 1e-8,
    max_eval_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Solve an MDP with policy iteration.

    Problem
    -------
    Policy iteration alternates between:
      1. Policy evaluation: run iterative_evaluation until convergence.
      2. Policy improvement: derive greedy policy from V.
    Stop when the policy does not change after improvement.

    Return the optimal value function, optimal policy, and the number of
    policy improvement steps (outer iterations).

    Hints
    -----
    - Reuse policy_evaluation_sweep in a loop until convergence.
    - Compare old and new policy via np.array_equal after improvement.
    - Start with a uniform random policy.

    Parameters
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition tensor.
    R : np.ndarray, shape (S, A, S)
        Reward tensor.
    gamma : float
        Discount factor.
    eval_theta : float
        Convergence threshold for policy evaluation.
    max_eval_iter : int
        Maximum evaluation sweeps per policy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (V_star, policy_star, n_policy_improvements)
    """
    # SOLUTION
    S, A, _ = P.shape
    policy = np.full((S, A), 1.0 / A)
    V = np.zeros(S)
    n_improvements = 0

    while True:
        # Policy evaluation until convergence
        for _ in range(max_eval_iter):
            V_new = policy_evaluation_sweep(V, P, R, policy, gamma)
            if np.max(np.abs(V_new - V)) < eval_theta:
                V = V_new
                break
            V = V_new

        # Policy improvement
        new_policy = greedy_policy_improvement(V, P, R, gamma)
        n_improvements += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return V, policy, n_improvements


def compare_convergence(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
) -> dict[str, int]:
    """
    Compare the number of outer iterations for policy iteration and value
    iteration on the same MDP.

    Problem
    -------
    Run both algorithms and return a dict with keys "policy_iteration" and
    "value_iteration", each mapping to the number of outer iterations
    (improvement steps for PI; sweeps for VI).

    Hints
    -----
    - Call value_iteration and policy_iteration defined above.
    - The returned counts are directly comparable as a rough indicator of
      which algorithm converges in fewer iterations (not wall-clock time).

    Parameters
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition tensor.
    R : np.ndarray, shape (S, A, S)
        Reward tensor.
    gamma : float
        Discount factor.

    Returns
    -------
    dict[str, int]
        {"policy_iteration": <int>, "value_iteration": <int>}
    """
    # SOLUTION
    _, _, pi_iters = policy_iteration(P, R, gamma)
    _, vi_iters = value_iteration(P, R, gamma)
    return {"policy_iteration": pi_iters, "value_iteration": vi_iters}


def test_exercise_4() -> None:
    # Use the same chain MDP as Exercise 3
    S, A = 3, 1
    P = np.zeros((S, A, S))
    P[0, 0, 1] = 1.0
    P[1, 0, 2] = 1.0
    P[2, 0, 2] = 1.0
    R = np.zeros((S, A, S))
    R[1, 0, 2] = 1.0
    gamma = 0.9

    counts = compare_convergence(P, R, gamma)
    assert "policy_iteration" in counts and "value_iteration" in counts, \
        "Result must have keys 'policy_iteration' and 'value_iteration'."
    assert isinstance(counts["policy_iteration"], int), \
        "policy_iteration count must be int."
    assert isinstance(counts["value_iteration"], int), \
        "value_iteration count must be int."
    # Policy iteration typically needs very few improvement steps
    assert counts["policy_iteration"] >= 1, \
        "At least 1 policy improvement step must occur."
    assert counts["value_iteration"] >= 1, \
        "At least 1 VI sweep must occur."

    # Both algorithms must agree on optimal values (within tolerance)
    V_pi, _, _ = policy_iteration(P, R, gamma)
    V_vi, _ = value_iteration(P, R, gamma)
    assert np.allclose(V_pi, V_vi, atol=1e-5), \
        f"PI and VI must produce the same V*. PI={V_pi}, VI={V_vi}."

    print("Exercise 4 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: One Sweep of Policy Evaluation", test_exercise_1),
        ("Exercise 2: Greedy Policy Improvement", test_exercise_2),
        ("Exercise 3: Value Iteration", test_exercise_3),
        ("Exercise 4: PI vs VI Convergence Comparison", test_exercise_4),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

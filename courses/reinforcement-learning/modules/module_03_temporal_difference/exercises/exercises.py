"""
Module 03 - Temporal Difference Learning: Self-Check Exercises

Covers:
- TD(0) state-value update
- Q-learning (off-policy TD) update
- SARSA (on-policy TD) update
- Classifying update rules as on-policy or off-policy

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: Single TD(0) Update Step
# ---------------------------------------------------------------------------

def td0_update(
    V: np.ndarray,
    state: int,
    reward: float,
    next_state: int,
    alpha: float,
    gamma: float,
    done: bool = False,
) -> np.ndarray:
    """
    Apply a single TD(0) update to the state-value function.

    Problem
    -------
    TD(0) bootstraps from the next state's value estimate:

        delta = r + gamma * V[s'] - V[s]     (if not done)
        delta = r - V[s]                      (if done / terminal)
        V[s] <- V[s] + alpha * delta

    The TD error delta measures the discrepancy between the current
    estimate V[s] and the TD target.

    Hints
    -----
    - When done=True, there is no next state; the target is just r.
    - Modify V in-place (or a copy) and return it.

    Parameters
    ----------
    V : np.ndarray, shape (S,)
        Current state-value estimates (modified in-place).
    state : int
        Current state s.
    reward : float
        Reward r received after taking the action.
    next_state : int
        Next state s' (ignored when done=True).
    alpha : float
        Learning rate in (0, 1].
    gamma : float
        Discount factor in [0, 1].
    done : bool
        Whether the episode terminated after this transition.

    Returns
    -------
    np.ndarray, shape (S,)
        Updated value estimates (same array, modified in-place).

    Examples
    --------
    >>> V = np.array([0.0, 0.0, 0.0])
    >>> td0_update(V, state=0, reward=1.0, next_state=1, alpha=0.1, gamma=0.9)
    array([0.1, 0.0, 0.0])  # delta=1+0.9*0-0=1; V[0] += 0.1*1
    """
    # SOLUTION
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * V[next_state]
    td_error = td_target - V[state]
    V[state] += alpha * td_error
    return V


def test_exercise_1() -> None:
    V = np.zeros(3)

    # Step from s=0, r=1, s'=1, not done
    # delta = 1 + 0.9*0 - 0 = 1; V[0] += 0.1*1 = 0.1
    V = td0_update(V, state=0, reward=1.0, next_state=1, alpha=0.1, gamma=0.9)
    assert np.isclose(V[0], 0.1), f"V[0] must be 0.1, got {V[0]}."
    assert V[1] == 0.0, "V[1] must be unchanged."

    # Terminal transition: target = r only
    V2 = np.array([5.0, 0.0])
    td0_update(V2, state=0, reward=1.0, next_state=1, alpha=0.5, gamma=0.9, done=True)
    # delta = 1 - 5 = -4; V[0] += 0.5 * -4 = -2; V[0] = 3
    assert np.isclose(V2[0], 3.0), f"Terminal update: V[0] must be 3.0, got {V2[0]}."

    # alpha=1: V converges in one step to target
    V3 = np.zeros(2)
    td0_update(V3, state=0, reward=2.0, next_state=1, alpha=1.0, gamma=0.0, done=False)
    # target = 2 + 0*0 = 2; V[0] = 0 + 1*(2-0) = 2
    assert np.isclose(V3[0], 2.0), f"alpha=1 must set V[s]=target, got {V3[0]}."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Single Q-Learning Update Step
# ---------------------------------------------------------------------------

def q_learning_update(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    alpha: float,
    gamma: float,
    done: bool = False,
) -> np.ndarray:
    """
    Apply a single Q-learning (off-policy TD) update.

    Problem
    -------
    Q-learning updates toward the greedy target, regardless of the
    policy used to collect the data:

        target = r + gamma * max_a' Q[s', a']   (if not done)
        target = r                                (if done)
        Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])

    Q-learning is off-policy because it learns the optimal Q-function
    even when the behavior policy is exploratory.

    Hints
    -----
    - Use np.max(Q[next_state]) for the greedy bootstrap.

    Parameters
    ----------
    Q : np.ndarray, shape (S, A)
        Current action-value estimates (modified in-place).
    state : int
        Current state s.
    action : int
        Action taken a.
    reward : float
        Reward received.
    next_state : int
        Next state s'.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    done : bool
        Whether the episode terminated.

    Returns
    -------
    np.ndarray, shape (S, A)
        Updated Q-table.
    """
    # SOLUTION
    if done:
        target = reward
    else:
        target = reward + gamma * np.max(Q[next_state])
    Q[state, action] += alpha * (target - Q[state, action])
    return Q


def test_exercise_2() -> None:
    Q = np.zeros((3, 2))

    # Update Q[0,1] with r=1, s'=1 (Q[1] = 0)
    # target = 1 + 0.9*max(0,0) = 1; Q[0,1] += 0.1*(1-0) = 0.1
    q_learning_update(Q, state=0, action=1, reward=1.0, next_state=1,
                      alpha=0.1, gamma=0.9)
    assert np.isclose(Q[0, 1], 0.1), f"Q[0,1] must be 0.1, got {Q[0,1]}."
    assert Q[0, 0] == 0.0, "Q[0,0] must be unchanged."

    # Set Q[1, 0]=5; now greedy target from s'=1 uses max=5
    Q2 = np.zeros((3, 2))
    Q2[1, 0] = 5.0
    q_learning_update(Q2, state=0, action=0, reward=0.0, next_state=1,
                      alpha=0.5, gamma=0.9)
    # target = 0 + 0.9*5 = 4.5; Q[0,0] += 0.5*(4.5-0) = 2.25
    assert np.isclose(Q2[0, 0], 2.25), f"Q[0,0] must be 2.25, got {Q2[0,0]}."

    # Terminal step
    Q3 = np.full((2, 2), 1.0)
    q_learning_update(Q3, state=0, action=0, reward=5.0, next_state=1,
                      alpha=1.0, gamma=0.9, done=True)
    # target = 5; Q[0,0] = 1 + 1*(5-1) = 5
    assert np.isclose(Q3[0, 0], 5.0), f"Terminal: Q[0,0] must be 5.0, got {Q3[0,0]}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Single SARSA Update Step
# ---------------------------------------------------------------------------

def sarsa_update(
    Q: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    next_action: int,
    alpha: float,
    gamma: float,
    done: bool = False,
) -> np.ndarray:
    """
    Apply a single SARSA (on-policy TD) update.

    Problem
    -------
    SARSA bootstraps from the Q-value of the *next action actually taken*
    under the behavior policy (State-Action-Reward-State-Action):

        target = r + gamma * Q[s', a']   (if not done)
        target = r                        (if done)
        Q[s, a] <- Q[s, a] + alpha * (target - Q[s, a])

    Unlike Q-learning, SARSA uses Q[s', a'] where a' is sampled from the
    behavior policy, making it on-policy.

    Hints
    -----
    - next_action is already sampled; simply look up Q[next_state, next_action].
    - Distinguish from Q-learning: SARSA uses the actual next action,
      not the greedy action.

    Parameters
    ----------
    Q : np.ndarray, shape (S, A)
        Current action-value estimates (modified in-place).
    state, action : int
        Current (s, a) pair.
    reward : float
        Reward received.
    next_state, next_action : int
        Next (s', a') pair sampled from behavior policy.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    done : bool
        Whether the episode terminated.

    Returns
    -------
    np.ndarray, shape (S, A)
        Updated Q-table.
    """
    # SOLUTION
    if done:
        target = reward
    else:
        target = reward + gamma * Q[next_state, next_action]
    Q[state, action] += alpha * (target - Q[state, action])
    return Q


def test_exercise_3() -> None:
    Q = np.zeros((3, 2))

    # Q[1,0]=2; SARSA uses next_action=0, so target=1+0.9*2=2.8
    Q[1, 0] = 2.0
    sarsa_update(Q, state=0, action=1, reward=1.0, next_state=1,
                 next_action=0, alpha=0.1, gamma=0.9)
    # Q[0,1] += 0.1*(2.8-0) = 0.28
    assert np.isclose(Q[0, 1], 0.28), f"Q[0,1] must be 0.28, got {Q[0,1]}."

    # Contrast with Q-learning: if next best action gives higher value
    # SARSA uses a'=0 (Q=2), while Q-learning would use max (also Q=2 here)
    Q_ql = np.zeros((3, 2))
    Q_ql[1, 0] = 2.0
    q_learning_update(Q_ql, state=0, action=1, reward=1.0, next_state=1,
                      alpha=0.1, gamma=0.9)
    # Both should be equal here since Q-greedy == next_action
    assert np.isclose(Q[0, 1], Q_ql[0, 1]), \
        "When next_action is greedy, SARSA and Q-learning must agree."

    # When next_action != greedy, SARSA gives different result
    Q2 = np.zeros((3, 2))
    Q2[1, 0] = 5.0  # greedy action a=0 has Q=5
    Q2[1, 1] = 1.0  # non-greedy action a=1 has Q=1
    sarsa_update(Q2, state=0, action=0, reward=0.0, next_state=1,
                 next_action=1, alpha=0.5, gamma=0.9)  # use non-greedy a'=1
    # SARSA target: 0 + 0.9*1 = 0.9; Q[0,0] += 0.5*(0.9-0)=0.45
    assert np.isclose(Q2[0, 0], 0.45), \
        f"SARSA with non-greedy next action: Q[0,0] must be 0.45, got {Q2[0,0]}."

    Q3 = np.zeros((3, 2))
    Q3[1, 0] = 5.0
    Q3[1, 1] = 1.0
    q_learning_update(Q3, state=0, action=0, reward=0.0, next_state=1,
                      alpha=0.5, gamma=0.9)  # uses max=5
    # Q-learning target: 0 + 0.9*5 = 4.5; Q[0,0] += 0.5*4.5=2.25
    assert np.isclose(Q3[0, 0], 2.25), \
        f"Q-learning with greedy max: Q[0,0] must be 2.25, got {Q3[0,0]}."
    # Verify they differ
    assert not np.isclose(Q2[0, 0], Q3[0, 0]), \
        "SARSA and Q-learning must differ when next_action is not greedy."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Exercise 4: Classify On-Policy vs Off-Policy
# ---------------------------------------------------------------------------

def classify_update_rule(rule_name: str) -> str:
    """
    Return whether the named update rule is "on-policy" or "off-policy".

    Problem
    -------
    Classify the following algorithms:
      - "sarsa"        : on-policy (bootstraps from the behavior policy's action)
      - "q_learning"   : off-policy (bootstraps from greedy max regardless of behavior)
      - "td0"          : on-policy (evaluates the behavior policy)
      - "expected_sarsa_off" : off-policy when target != behavior policy
      - "double_q_learning"  : off-policy (decoupled greedy selection)

    Hints
    -----
    - On-policy: the update target depends on the *same* policy used to
      collect data.
    - Off-policy: the update target uses a *different* (often greedy)
      policy than the one generating trajectories.

    Parameters
    ----------
    rule_name : str
        One of: "sarsa", "q_learning", "td0",
                "expected_sarsa_off", "double_q_learning".

    Returns
    -------
    str
        "on-policy" or "off-policy".

    Raises
    ------
    ValueError
        If rule_name is not recognised.
    """
    # SOLUTION
    ON_POLICY = {"sarsa", "td0"}
    OFF_POLICY = {"q_learning", "expected_sarsa_off", "double_q_learning"}
    if rule_name in ON_POLICY:
        return "on-policy"
    if rule_name in OFF_POLICY:
        return "off-policy"
    raise ValueError(f"Unknown rule: {rule_name!r}")


def test_exercise_4() -> None:
    on_policy_rules = ["sarsa", "td0"]
    off_policy_rules = ["q_learning", "expected_sarsa_off", "double_q_learning"]

    for rule in on_policy_rules:
        result = classify_update_rule(rule)
        assert result == "on-policy", \
            f"{rule!r} must be 'on-policy', got {result!r}."

    for rule in off_policy_rules:
        result = classify_update_rule(rule)
        assert result == "off-policy", \
            f"{rule!r} must be 'off-policy', got {result!r}."

    # Unknown rule raises ValueError
    raised = False
    try:
        classify_update_rule("unknown_algo")
    except ValueError:
        raised = True
    assert raised, "Unknown rule name must raise ValueError."

    print("Exercise 4 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: TD(0) Update", test_exercise_1),
        ("Exercise 2: Q-Learning Update", test_exercise_2),
        ("Exercise 3: SARSA Update", test_exercise_3),
        ("Exercise 4: On-Policy vs Off-Policy Classification", test_exercise_4),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

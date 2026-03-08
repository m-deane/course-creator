"""
Module 09 - Frontiers of Reinforcement Learning: Self-Check Exercises

Covers:
- Preference-based reward model (RLHF concept)
- Conservative Q-Learning (CQL) penalty on Q-values
- Discrete trading environment step function

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: Preference-Based Reward Model
# ---------------------------------------------------------------------------

class PreferenceRewardModel:
    """
    Simple reward model trained on pairwise human preferences (RLHF concept).

    Problem
    -------
    In Reinforcement Learning from Human Feedback (RLHF), a reward model
    r_phi is trained to assign scalar rewards to trajectories such that
    preferred trajectories receive higher rewards.

    This exercise implements a linear reward model:
        r_phi(trajectory) = phi(trajectory)^T * weights

    where phi is a hand-crafted feature function and weights are updated
    via preference comparisons.

    Training signal: given two trajectories (A, B) where humans prefer A,
    update weights so that r(A) > r(B) using a simple perceptron-style update:

        If r(A) <= r(B): weights += lr * (phi(A) - phi(B))

    Implement:
      - predict(features): scalar reward = features @ weights.
      - update(features_preferred, features_rejected, lr): update weights.
      - accuracy(comparisons): fraction of pairs where preferred > rejected.

    Hints
    -----
    - Initialize weights to zeros.
    - The update only fires on incorrect or tied predictions.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of trajectory feature vectors.

    Examples
    --------
    >>> model = PreferenceRewardModel(feature_dim=3)
    >>> model.predict(np.array([1.0, 0.0, 0.0]))
    0.0   # zero weights initially
    """

    def __init__(self, feature_dim: int) -> None:
        # SOLUTION
        self.weights = np.zeros(feature_dim)

    def predict(self, features: np.ndarray) -> float:
        """
        Compute the scalar reward for a trajectory's feature vector.

        Parameters
        ----------
        features : np.ndarray, shape (D,)
            Feature vector of a trajectory.

        Returns
        -------
        float
            Predicted reward scalar.
        """
        # SOLUTION
        return float(self.weights @ features)

    def update(
        self,
        features_preferred: np.ndarray,
        features_rejected: np.ndarray,
        lr: float = 0.01,
    ) -> None:
        """
        Update weights so that r(preferred) > r(rejected).

        Problem
        -------
        If the model currently ranks preferred <= rejected, update:
            weights += lr * (features_preferred - features_rejected)

        No update is made when the preferred trajectory is already ranked higher.

        Parameters
        ----------
        features_preferred : np.ndarray, shape (D,)
            Feature vector of the human-preferred trajectory.
        features_rejected : np.ndarray, shape (D,)
            Feature vector of the human-rejected trajectory.
        lr : float
            Learning rate.
        """
        # SOLUTION
        r_pref = self.predict(features_preferred)
        r_rej = self.predict(features_rejected)
        if r_pref <= r_rej:
            self.weights += lr * (features_preferred - features_rejected)

    def accuracy(
        self,
        comparisons: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """
        Fraction of (preferred, rejected) pairs correctly ranked.

        Parameters
        ----------
        comparisons : list of (preferred_features, rejected_features)
            List of pairwise comparisons.

        Returns
        -------
        float
            Fraction in [0, 1] of pairs where r(preferred) > r(rejected).
        """
        # SOLUTION
        if not comparisons:
            return 0.0
        correct = sum(
            self.predict(pref) > self.predict(rej)
            for pref, rej in comparisons
        )
        return correct / len(comparisons)


def test_exercise_1() -> None:
    model = PreferenceRewardModel(feature_dim=3)

    # Initial prediction is 0
    assert model.predict(np.array([1.0, 2.0, 3.0])) == 0.0, \
        "Initial prediction must be 0 for zero weights."

    # After many updates, preferred trajectory should rank higher
    pref = np.array([1.0, 0.0, 0.0])
    rej = np.array([0.0, 1.0, 0.0])
    for _ in range(50):
        model.update(pref, rej, lr=0.1)

    assert model.predict(pref) > model.predict(rej), \
        "After training, preferred trajectory must have higher reward."

    # Accuracy on training pair
    acc = model.accuracy([(pref, rej)])
    assert acc == 1.0, f"Accuracy on training pair must be 1.0, got {acc}."

    # No update when already correct
    w_before = model.weights.copy()
    model.update(pref, rej, lr=0.1)  # already correct, no update
    assert np.allclose(model.weights, w_before), \
        "No weight update should occur when model already ranks correctly."

    # Empty comparisons
    assert model.accuracy([]) == 0.0, \
        "Empty comparisons must return accuracy 0.0."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Conservative Q-Learning (CQL) Penalty
# ---------------------------------------------------------------------------

def cql_penalty(
    Q: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute the CQL regularisation penalty for offline RL.

    Problem
    -------
    Conservative Q-Learning (CQL) addresses the overestimation problem in
    offline RL by penalising Q-values for out-of-distribution actions.

    The CQL penalty term (simplified tabular form) is:

        CQL_penalty = alpha * mean_s [ logsumexp_a(Q[s, :]) - Q[s, a_dataset] ]

    where:
      - logsumexp_a(Q[s, :]) represents the soft-maximum over all actions
        (encourages Q to be low for unobserved actions).
      - Q[s, a_dataset] rewards the data-covered (s, a) pairs.
      - alpha controls the strength of conservatism.

    Compute this penalty over the provided (states, actions) batch.

    Hints
    -----
    - scipy.special.logsumexp or implement manually:
      logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
    - Compute per-state logsumexp on Q[s, :], then subtract Q[s, a_dataset].
    - Take mean over the batch.

    Parameters
    ----------
    Q : np.ndarray, shape (S, A)
        Current Q-table.
    states : np.ndarray, shape (B,), dtype int
        Batch of states from the offline dataset.
    actions : np.ndarray, shape (B,), dtype int
        Corresponding actions from the offline dataset.
    alpha : float
        Penalty coefficient.

    Returns
    -------
    float
        CQL penalty value (non-negative).

    Examples
    --------
    >>> Q = np.array([[1.0, 2.0], [0.0, 3.0]])
    >>> cql_penalty(Q, np.array([0, 1]), np.array([0, 1]), alpha=1.0)
    # logsumexp([1,2])=2.31, Q[0,0]=1 -> diff=1.31
    # logsumexp([0,3])=3.05, Q[1,1]=3 -> diff=0.05
    # mean(1.31, 0.05) = 0.68  (approximate)
    """
    # SOLUTION
    def _logsumexp(x: np.ndarray) -> float:
        m = np.max(x)
        return m + np.log(np.sum(np.exp(x - m)))

    penalties = []
    for s, a in zip(states, actions):
        lse = _logsumexp(Q[s, :])
        q_data = Q[s, a]
        penalties.append(lse - q_data)

    return float(alpha * np.mean(penalties))


def test_exercise_2() -> None:
    # Simple 2-state, 2-action Q-table
    Q = np.array([[1.0, 2.0], [0.0, 3.0]])
    states = np.array([0, 1])
    actions = np.array([0, 1])

    penalty = cql_penalty(Q, states, actions, alpha=1.0)
    assert isinstance(penalty, float), f"Penalty must be float, got {type(penalty)}."
    assert penalty >= 0.0, f"CQL penalty must be non-negative, got {penalty}."

    # Verify approximate value
    # logsumexp([1,2]) = 2 + log(1+exp(-1)) ~ 2.3133
    # logsumexp([0,3]) = 3 + log(exp(-3)+1) ~ 3.0486
    # diffs: 2.3133-1=1.3133, 3.0486-3=0.0486; mean ~ 0.681
    assert abs(penalty - 0.681) < 0.01, \
        f"CQL penalty must be ~0.681, got {penalty:.4f}."

    # alpha=0: penalty must be 0
    assert cql_penalty(Q, states, actions, alpha=0.0) == 0.0, \
        "alpha=0 must give penalty=0."

    # Higher alpha -> proportionally higher penalty
    penalty_2x = cql_penalty(Q, states, actions, alpha=2.0)
    assert np.isclose(penalty_2x, 2 * penalty, rtol=1e-5), \
        "Doubling alpha must double the penalty."

    # When dataset action is already the greedy action (argmax),
    # the logsumexp - Q[s,a] is minimised
    Q2 = np.array([[0.0, 5.0]])  # best action is a=1
    p_greedy = cql_penalty(Q2, np.array([0]), np.array([1]), alpha=1.0)
    p_non_greedy = cql_penalty(Q2, np.array([0]), np.array([0]), alpha=1.0)
    assert p_greedy < p_non_greedy, \
        "CQL penalty must be lower when dataset action is optimal."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Discrete Trading Environment Step Function
# ---------------------------------------------------------------------------

class TradingEnv:
    """
    Minimal discrete trading environment for RL research.

    Problem
    -------
    A trading environment where an agent decides at each step to:
      - Action 0: Hold (do nothing)
      - Action 1: Buy  (go long if not already, or ignore if already long)
      - Action 2: Sell (go flat if long, or ignore if already flat)

    State: (position, price_index)
      - position: 0 = flat, 1 = long
      - price_index: integer index into the price series

    Reward:
      - After BUY at price p_t: reward = 0 (no reward on entry)
      - After SELL (close long) at price p_t: reward = p_t - p_entry
        (profit = current price - entry price)
      - After HOLD: reward = 0
      - Episode ends when price_index >= len(prices) - 1

    Implement:
      - reset(): return initial state (position=0, price_index=0).
      - step(action): return (next_state, reward, done).

    Hints
    -----
    - Store entry_price when a BUY occurs.
    - On SELL (action=2) with position=1: reward = current_price - entry_price.
    - done=True when price_index reaches the last price.

    Parameters
    ----------
    prices : np.ndarray, shape (T,)
        Historical price series.
    """

    def __init__(self, prices: np.ndarray) -> None:
        # SOLUTION
        self.prices = prices
        self.position = 0
        self.price_index = 0
        self.entry_price = 0.0

    def reset(self) -> tuple[int, int]:
        """
        Reset environment to initial state.

        Returns
        -------
        tuple[int, int]
            (position, price_index) = (0, 0)
        """
        # SOLUTION
        self.position = 0
        self.price_index = 0
        self.entry_price = 0.0
        return (self.position, self.price_index)

    def step(self, action: int) -> tuple[tuple[int, int], float, bool]:
        """
        Take one step in the trading environment.

        Problem
        -------
        Execute the action and advance the price index by 1:
          - Action 0 (Hold): reward = 0, position unchanged.
          - Action 1 (Buy): if flat, go long; record entry_price.
          - Action 2 (Sell): if long, go flat; reward = price - entry_price.

        Parameters
        ----------
        action : int
            0=Hold, 1=Buy, 2=Sell.

        Returns
        -------
        tuple[tuple[int,int], float, bool]
            (next_state, reward, done)
            next_state = (position, new_price_index)
            done = True if new_price_index >= len(prices) - 1
        """
        # SOLUTION
        current_price = self.prices[self.price_index]
        reward = 0.0

        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 1:  # Sell
            reward = float(current_price - self.entry_price)
            self.position = 0

        self.price_index = min(self.price_index + 1, len(self.prices) - 1)
        done = self.price_index >= len(self.prices) - 1
        next_state = (self.position, self.price_index)
        return next_state, reward, done


def test_exercise_3() -> None:
    prices = np.array([100.0, 105.0, 102.0, 110.0])
    env = TradingEnv(prices)

    # Reset
    state = env.reset()
    assert state == (0, 0), f"Initial state must be (0, 0), got {state}."

    # Hold: no position change, no reward
    next_state, reward, done = env.step(0)
    assert next_state == (0, 1), f"After Hold: state must be (0,1), got {next_state}."
    assert reward == 0.0, f"Hold must give 0 reward, got {reward}."
    assert not done, "Not done after step 1 of 4-price series."

    # Buy at price[1]=105
    next_state, reward, done = env.step(1)
    assert next_state == (1, 2), f"After Buy: state must be (1,2), got {next_state}."
    assert reward == 0.0, "Buy must give 0 reward."
    assert env.entry_price == 105.0, \
        f"entry_price must be 105.0, got {env.entry_price}."

    # Sell at price[2]=102: profit = 102 - 105 = -3
    next_state, reward, done = env.step(2)
    assert next_state[0] == 0, "After Sell: position must be flat (0)."
    assert np.isclose(reward, -3.0), \
        f"Sell at 102 after buy at 105 gives profit -3, got {reward}."

    # Buy again at price[3]=110, episode ends at last price
    next_state, reward, done = env.step(1)
    assert done, "After step 4 of 4-price series, done must be True."

    # Hold when flat: no reward, position stays 0
    env2 = TradingEnv(prices)
    env2.reset()
    ns, r, _ = env2.step(2)  # Sell when flat: ignored
    assert r == 0.0, "Sell when flat must give 0 reward."
    assert ns[0] == 0, "Sell when flat must not change position."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: Preference-Based Reward Model", test_exercise_1),
        ("Exercise 2: CQL Conservative Penalty", test_exercise_2),
        ("Exercise 3: Trading Environment Step Function", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

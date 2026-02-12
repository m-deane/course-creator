"""
Module 6 Exercises: Advanced Topics in Multi-Armed Bandits

Self-check exercises with assert-based validation.
Complete all exercises to master non-stationary, restless, and adversarial bandits.
"""

import numpy as np
from scipy.stats import beta
from collections import deque


# ============================================================================
# Exercise 1: Implement Discounted Thompson Sampling
# ============================================================================

class DiscountedThompsonSampling:
    """
    TODO: Implement Discounted Thompson Sampling with configurable decay factor.

    The algorithm should:
    1. Maintain alpha and beta parameters for each arm
    2. Discount ALL arms' parameters by gamma each round (exponential forgetting)
    3. Update selected arm with observed reward
    4. Sample from Beta distributions to select arms
    """

    def __init__(self, n_arms, gamma=0.95):
        """
        Initialize Discounted Thompson Sampling.

        Args:
            n_arms: Number of arms
            gamma: Discount factor (0 < gamma < 1). Higher = slower forgetting.
        """
        self.n_arms = n_arms
        self.gamma = gamma
        # TODO: Initialize alpha and beta parameters (start with 1.0 for each arm)
        self.alpha = None  # Replace with proper initialization
        self.beta_params = None  # Replace with proper initialization

    def select_arm(self):
        """
        TODO: Sample from Beta(alpha, beta) for each arm and return argmax.

        Returns:
            int: Index of selected arm
        """
        # TODO: Implement Thompson sampling selection
        pass

    def update(self, arm, reward):
        """
        TODO: Update parameters with discounting.

        1. Multiply ALL alpha and beta parameters by gamma (discount old data)
        2. Add reward to alpha[arm]
        3. Add (1 - reward) to beta[arm]

        Args:
            arm: Selected arm index
            reward: Observed reward (0 or 1 for Bernoulli)
        """
        # TODO: Implement discounted update
        pass


# Test Exercise 1
def test_exercise_1():
    """Test Discounted Thompson Sampling implementation."""
    np.random.seed(42)

    bandit = DiscountedThompsonSampling(n_arms=3, gamma=0.95)

    # Check initialization
    assert bandit.alpha is not None, "Initialize alpha parameters"
    assert bandit.beta_params is not None, "Initialize beta parameters"
    assert len(bandit.alpha) == 3, "Alpha should have length n_arms"
    assert len(bandit.beta_params) == 3, "Beta should have length n_arms"

    # Test selection
    arm = bandit.select_arm()
    assert isinstance(arm, (int, np.integer)), "select_arm should return integer"
    assert 0 <= arm < 3, "Selected arm should be in valid range"

    # Test update with reward=1
    initial_alpha = bandit.alpha.copy()
    initial_beta = bandit.beta_params.copy()

    bandit.update(arm, reward=1)

    # Check discounting happened
    for i in range(3):
        if i == arm:
            # Selected arm: discounted + incremented
            assert bandit.alpha[i] > initial_alpha[i] * bandit.gamma, \
                f"Alpha[{i}] should increase (discounted + reward)"
        else:
            # Unselected arms: only discounted
            assert abs(bandit.alpha[i] - initial_alpha[i] * bandit.gamma) < 1e-6, \
                f"Alpha[{i}] should only be discounted"

    # Test non-stationarity: run on switching bandit
    def get_probs(t):
        return [0.7, 0.3, 0.3] if t < 50 else [0.3, 0.7, 0.3]

    bandit = DiscountedThompsonSampling(n_arms=3, gamma=0.95)
    selections = []

    for t in range(100):
        arm = bandit.select_arm()
        selections.append(arm)
        probs = get_probs(t)
        reward = np.random.binomial(1, probs[arm])
        bandit.update(arm, reward)

    # After regime change, should switch to arm 1
    later_selections = selections[70:]  # After transition period
    arm1_freq = later_selections.count(1) / len(later_selections)
    assert arm1_freq > 0.5, "Should adapt to new best arm (arm 1) after regime shift"

    print("✓ Exercise 1 passed: Discounted Thompson Sampling works!")


# ============================================================================
# Exercise 2: Build CUSUM-Based Bandit Restarter
# ============================================================================

class CUSUMDetector:
    """
    CUSUM change-point detector.
    Signals when cumulative deviations exceed threshold.
    """

    def __init__(self, mu0=0.5, k=0.1, h=5.0):
        """
        Args:
            mu0: Baseline mean (pre-change)
            k: Slack parameter (tolerance for noise)
            h: Threshold for detection
        """
        self.mu0 = mu0
        self.k = k
        self.h = h
        self.S = 0

    def update(self, x):
        """Update CUSUM and check for change."""
        self.S = max(0, self.S + (x - self.mu0) - self.k)
        return self.S > self.h

    def reset(self):
        """Reset after change detected."""
        self.S = 0


class ThompsonSamplingWithCUSUM:
    """
    TODO: Implement Thompson Sampling with CUSUM-based change detection.

    When CUSUM detects a change:
    1. Reset alpha and beta parameters to 1.0 (restart exploration)
    2. Reset all CUSUM detectors
    3. Record the change point
    """

    def __init__(self, n_arms, cusum_k=0.1, cusum_h=5.0):
        """
        Initialize Thompson Sampling with CUSUM.

        Args:
            n_arms: Number of arms
            cusum_k: CUSUM slack parameter
            cusum_h: CUSUM detection threshold
        """
        self.n_arms = n_arms
        # TODO: Initialize alpha and beta parameters
        self.alpha = None
        self.beta_params = None

        # TODO: Create one CUSUM detector per arm
        self.detectors = None

        # Track detected change points
        self.change_points = []

    def select_arm(self):
        """
        TODO: Implement Thompson sampling selection.

        Returns:
            int: Selected arm index
        """
        pass

    def update(self, arm, reward, t):
        """
        TODO: Update bandit and check for regime change.

        Steps:
        1. Update alpha and beta for selected arm (standard Thompson update)
        2. Update CUSUM detector for selected arm
        3. If change detected:
           - Reset all alpha and beta to 1.0
           - Reset all CUSUM detectors
           - Record change point

        Args:
            arm: Selected arm
            reward: Observed reward
            t: Current time step

        Returns:
            bool: True if change detected, False otherwise
        """
        pass


# Test Exercise 2
def test_exercise_2():
    """Test CUSUM-based bandit restarter."""
    np.random.seed(42)

    # Create bandit with regime changes at t=100 and t=200
    def get_probs(t):
        if t < 100:
            return [0.7, 0.3, 0.3]
        elif t < 200:
            return [0.3, 0.7, 0.3]
        else:
            return [0.3, 0.3, 0.7]

    bandit = ThompsonSamplingWithCUSUM(n_arms=3, cusum_k=0.05, cusum_h=4.0)

    # Check initialization
    assert bandit.alpha is not None, "Initialize alpha"
    assert bandit.beta_params is not None, "Initialize beta"
    assert bandit.detectors is not None, "Initialize CUSUM detectors"
    assert len(bandit.detectors) == 3, "One detector per arm"

    # Run bandit
    for t in range(300):
        arm = bandit.select_arm()
        probs = get_probs(t)
        reward = np.random.binomial(1, probs[arm])
        change_detected = bandit.update(arm, reward, t)

    # Should detect at least one change (likely both)
    assert len(bandit.change_points) >= 1, \
        "Should detect at least one regime change"

    # Change points should be reasonably close to true changes (within ~50 steps)
    true_changes = [100, 200]
    for true_change in true_changes:
        close_detections = [cp for cp in bandit.change_points
                            if abs(cp - true_change) < 50]
        assert len(close_detections) > 0 or len(bandit.change_points) == 0, \
            f"Should detect change near t={true_change} (within 50 steps)"

    print("✓ Exercise 2 passed: CUSUM-based restarter works!")


# ============================================================================
# Exercise 3: Compare Standard vs Non-Stationary on Seasonal Pattern
# ============================================================================

def seasonal_reward_probabilities(t, period=100):
    """
    Seasonal commodity pattern: rewards oscillate with period.

    Args:
        t: Time step
        period: Period of seasonality

    Returns:
        list: Reward probabilities for 3 arms
    """
    phase = (t % period) / period  # Normalize to [0, 1]

    # Arm 0: High in first half, low in second half (e.g., heating oil in winter)
    p0 = 0.7 if phase < 0.5 else 0.3

    # Arm 1: High in second half (e.g., cooling energy in summer)
    p1 = 0.3 if phase < 0.5 else 0.7

    # Arm 2: Stable year-round
    p2 = 0.5

    return [p0, p1, p2]


def compare_algorithms_on_seasonal_pattern():
    """
    TODO: Compare standard vs non-stationary bandits on seasonal commodity pattern.

    Steps:
    1. Implement StandardThompsonSampling (no discounting)
    2. Run both StandardThompsonSampling and DiscountedThompsonSampling
       for T=500 steps on seasonal_reward_probabilities
    3. Compute cumulative regret for each (regret = optimal - actual)
    4. Return regret difference (standard - discounted)

    Returns:
        float: Regret difference (should be positive if discounted is better)
    """
    np.random.seed(42)
    T = 500

    # TODO: Implement comparison
    # Hint: Track rewards for each algorithm, compute regret vs oracle
    # Oracle selects best arm each period based on seasonal_reward_probabilities

    pass


# Test Exercise 3
def test_exercise_3():
    """Test seasonal pattern comparison."""
    regret_difference = compare_algorithms_on_seasonal_pattern()

    assert regret_difference is not None, "Implement compare_algorithms_on_seasonal_pattern"
    assert isinstance(regret_difference, (int, float)), "Should return numeric regret difference"
    assert regret_difference > 0, \
        "Discounted TS should have lower regret than standard TS on seasonal pattern"

    print(f"✓ Exercise 3 passed: Discounted TS beats standard TS by {regret_difference:.1f} regret!")


# ============================================================================
# Bonus Exercise: EXP3 Implementation
# ============================================================================

class EXP3:
    """
    TODO (BONUS): Implement EXP3 for adversarial bandits.

    Algorithm:
    1. Maintain weights w_i for each arm
    2. Convert weights to probabilities with exploration: p_i = (1-γ)*w_i/Σw_j + γ/K
    3. Sample arm according to probabilities
    4. Update weights: w_i *= exp(γ * r̂_i / K) where r̂_i = r_i / p_i (importance sampling)
    """

    def __init__(self, n_arms, gamma=0.1):
        """
        Initialize EXP3.

        Args:
            n_arms: Number of arms
            gamma: Exploration parameter
        """
        self.n_arms = n_arms
        self.gamma = gamma
        # TODO: Initialize weights (all equal initially)
        self.weights = None

    def get_probabilities(self):
        """
        TODO: Compute selection probabilities from weights.

        Formula: p_i = (1-γ) * w_i / Σw_j + γ/K

        Returns:
            np.array: Selection probabilities
        """
        pass

    def select_arm(self):
        """
        TODO: Sample arm according to probabilities.

        Returns:
            int: Selected arm
        """
        pass

    def update(self, arm, reward):
        """
        TODO: Update weights using multiplicative rule.

        Steps:
        1. Compute estimated reward: r̂ = reward / p_arm (importance sampling)
        2. Update weight: w_arm *= exp(γ * r̂ / K)

        Args:
            arm: Selected arm
            reward: Observed reward in [0, 1]
        """
        pass


# Test Bonus Exercise
def test_bonus_exp3():
    """Test EXP3 implementation."""
    np.random.seed(42)

    bandit = EXP3(n_arms=3, gamma=0.1)

    # Check initialization
    assert bandit.weights is not None, "Initialize weights"
    assert len(bandit.weights) == 3, "Weights for all arms"

    # Test probability computation
    probs = bandit.get_probabilities()
    assert probs is not None, "Implement get_probabilities"
    assert len(probs) == 3, "Probability for each arm"
    assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities should sum to 1"

    # Test selection
    arm = bandit.select_arm()
    assert isinstance(arm, (int, np.integer)), "Should return integer"
    assert 0 <= arm < 3, "Valid arm index"

    # Test update
    initial_weights = bandit.weights.copy()
    bandit.update(arm, reward=1.0)
    assert not np.array_equal(bandit.weights, initial_weights), \
        "Weights should change after update"

    # Run on adversarial scenario (rewards depend on frequency of selection)
    bandit = EXP3(n_arms=3, gamma=0.2)
    counts = np.zeros(3)

    for t in range(200):
        arm = bandit.select_arm()
        counts[arm] += 1

        # Adversarial: arm becomes worse as it's selected more
        reward = 1.0 - (counts[arm] / 100)  # Decreasing reward
        reward = max(0, min(1, reward))  # Clip to [0, 1]

        bandit.update(arm, reward)

    # EXP3 should spread selections (not concentrate on one arm)
    min_selection_freq = counts.min() / counts.sum()
    assert min_selection_freq > 0.15, \
        "EXP3 should explore all arms (no arm < 15% frequency)"

    print("✓ Bonus Exercise passed: EXP3 works!")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("Module 6 Exercises: Advanced Topics\n")

    try:
        print("Exercise 1: Discounted Thompson Sampling")
        test_exercise_1()
        print()
    except (AssertionError, NotImplementedError) as e:
        print(f"✗ Exercise 1 failed: {e}\n")

    try:
        print("Exercise 2: CUSUM-Based Bandit Restarter")
        test_exercise_2()
        print()
    except (AssertionError, NotImplementedError) as e:
        print(f"✗ Exercise 2 failed: {e}\n")

    try:
        print("Exercise 3: Seasonal Pattern Comparison")
        test_exercise_3()
        print()
    except (AssertionError, NotImplementedError) as e:
        print(f"✗ Exercise 3 failed: {e}\n")

    try:
        print("Bonus Exercise: EXP3 for Adversarial Bandits")
        test_bonus_exp3()
        print()
    except (AssertionError, NotImplementedError) as e:
        print(f"✗ Bonus Exercise failed: {e}\n")

    print("=" * 60)
    print("Exercises complete! Review any failures above.")
    print("=" * 60)

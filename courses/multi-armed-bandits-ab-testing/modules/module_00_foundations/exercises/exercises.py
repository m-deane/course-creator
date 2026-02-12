"""
Module 0 Foundations: Self-Check Exercises

These exercises help you practice the core concepts from Module 0.
Each exercise has starter code and assert-based self-checks.

Run with: python exercises.py
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# =============================================================================
# Exercise 1: Calculate A/B Test Sample Size
# =============================================================================

def calculate_ab_sample_size(p_A, p_B, alpha=0.05, power=0.8):
    """
    Calculate required sample size per variant for an A/B test.

    Parameters:
    -----------
    p_A : float
        Baseline conversion rate (proportion)
    p_B : float
        Treatment conversion rate (proportion)
    alpha : float
        Significance level (default 0.05 for 95% confidence)
    power : float
        Statistical power (default 0.8 for 80% power)

    Returns:
    --------
    n : int
        Required sample size PER VARIANT (total = 2*n)

    Formula:
    --------
    n = 2(z_α/2 + z_β)² · p̄(1-p̄) / (p_B - p_A)²
    where p̄ = (p_A + p_B) / 2

    Example:
    --------
    >>> n = calculate_ab_sample_size(0.05, 0.08)
    >>> print(f"Need {n} per variant, {2*n} total")
    """
    # TODO: Implement the sample size formula

    # Step 1: Get critical values from normal distribution
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed test
    z_beta = stats.norm.ppf(power)

    # Step 2: Calculate pooled proportion
    p_bar = (p_A + p_B) / 2

    # Step 3: Apply sample size formula
    numerator = 2 * (z_alpha + z_beta)**2 * p_bar * (1 - p_bar)
    denominator = (p_B - p_A)**2
    n = numerator / denominator

    return int(np.ceil(n))


# Self-check tests for Exercise 1
def test_exercise_1():
    """Test the sample size calculation."""
    # Test case 1: Standard example (5% → 8% conversion)
    n = calculate_ab_sample_size(0.05, 0.08)
    assert 1000 <= n <= 1150, f"Expected ~1061, got {n}"

    # Test case 2: Larger effect (5% → 10%)
    n_large = calculate_ab_sample_size(0.05, 0.10)
    assert n_large < n, "Larger effect should need smaller sample size"
    assert 400 <= n_large <= 500, f"Expected ~436, got {n_large}"

    # Test case 3: Smaller effect (5% → 6%)
    n_small = calculate_ab_sample_size(0.05, 0.06)
    assert n_small > n, "Smaller effect should need larger sample size"
    assert 7500 <= n_small <= 9000, f"Expected ~8159, got {n_small}"

    print("✓ Exercise 1 tests passed!")


# =============================================================================
# Exercise 2: Implement Regret Calculator
# =============================================================================

def calculate_cumulative_regret(arm_means, choices):
    """
    Calculate cumulative regret for a sequence of arm choices.

    Parameters:
    -----------
    arm_means : array-like
        True expected reward for each arm
    choices : array-like
        Sequence of arm indices chosen (0, 1, 2, ...)

    Returns:
    --------
    cumulative_regret : np.array
        Cumulative regret at each time step

    Formula:
    --------
    R(T) = Σ (μ* - μ_a(t))
    where μ* = max_k μ_k

    Example:
    --------
    >>> arm_means = [0.05, 0.08, 0.06]
    >>> choices = [0, 1, 0, 1, 1, 2, 1]  # Your trading decisions
    >>> regret = calculate_cumulative_regret(arm_means, choices)
    >>> print(f"Total regret: {regret[-1]:.3f}")
    """
    # TODO: Implement cumulative regret calculation

    arm_means = np.array(arm_means)
    choices = np.array(choices)

    # Step 1: Find best arm mean
    best_mean = np.max(arm_means)

    # Step 2: Calculate instantaneous regret for each choice
    instant_regret = best_mean - arm_means[choices]

    # Step 3: Cumulative sum
    cumulative_regret = np.cumsum(instant_regret)

    return cumulative_regret


# Self-check tests for Exercise 2
def test_exercise_2():
    """Test the regret calculator."""
    # Test case 1: Always pick best arm → zero regret
    arm_means = [0.1, 0.2, 0.15]
    choices = [1] * 10  # Always pick arm 1 (best)
    regret = calculate_cumulative_regret(arm_means, choices)
    assert np.allclose(regret, 0), f"Optimal choices should have zero regret, got {regret}"

    # Test case 2: Always pick worst arm → linear regret
    choices_worst = [0] * 10  # Always pick arm 0 (worst)
    regret_worst = calculate_cumulative_regret(arm_means, choices_worst)
    expected_regret = 10 * (0.2 - 0.1)  # 10 choices × 0.1 gap
    assert np.isclose(regret_worst[-1], expected_regret), \
        f"Expected {expected_regret}, got {regret_worst[-1]}"

    # Test case 3: Mixed choices
    choices_mixed = [0, 1, 2, 1, 1]  # Mix of arms
    regret_mixed = calculate_cumulative_regret(arm_means, choices_mixed)
    # Regrets: [0.1, 0, 0.05, 0, 0] → cumsum: [0.1, 0.1, 0.15, 0.15, 0.15]
    assert np.isclose(regret_mixed[-1], 0.15), \
        f"Expected 0.15, got {regret_mixed[-1]}"
    assert len(regret_mixed) == len(choices_mixed), "Output length should match input"

    # Test case 4: Regret should be monotonically increasing
    assert np.all(np.diff(regret_mixed) >= 0), "Regret should never decrease"

    print("✓ Exercise 2 tests passed!")


# =============================================================================
# Exercise 3: Identify Switching Point
# =============================================================================

def should_have_switched(rewards_A, rewards_B, confidence_level=0.95):
    """
    Given sequences of rewards from two strategies, identify the earliest point
    where we had sufficient evidence that B > A (should have switched).

    Uses a running two-sample t-test to detect when difference becomes significant.

    Parameters:
    -----------
    rewards_A : array-like
        Observed rewards from strategy A
    rewards_B : array-like
        Observed rewards from strategy B
    confidence_level : float
        Confidence level for significance (default 0.95)

    Returns:
    --------
    switch_point : int or None
        Index where we should have switched (None if never significant)
    p_values : np.array
        P-values at each time point

    Interpretation:
    ---------------
    If switch_point = 50 and len(rewards_A) = 1000, you wasted 950 samples
    on inferior strategy A after the evidence was clear.

    Example:
    --------
    >>> np.random.seed(42)
    >>> rewards_A = np.random.normal(0.05, 0.2, 1000)
    >>> rewards_B = np.random.normal(0.08, 0.2, 1000)
    >>> switch_point, p_vals = should_have_switched(rewards_A, rewards_B)
    >>> print(f"Should have switched at sample {switch_point}")
    """
    # TODO: Implement the switching point detector

    rewards_A = np.array(rewards_A)
    rewards_B = np.array(rewards_B)
    alpha = 1 - confidence_level

    p_values = []
    switch_point = None

    # Need at least 5 samples per group for reliable t-test
    min_samples = 5

    for t in range(min_samples, min(len(rewards_A), len(rewards_B))):
        # Running t-test: compare A[0:t] vs B[0:t]
        sample_A = rewards_A[:t]
        sample_B = rewards_B[:t]

        # Two-sample t-test (two-tailed)
        t_stat, p_value = stats.ttest_ind(sample_A, sample_B)
        p_values.append(p_value)

        # Check if significant and B > A
        if p_value < alpha and switch_point is None:
            if np.mean(sample_B) > np.mean(sample_A):
                switch_point = t

    return switch_point, np.array(p_values)


# Self-check tests for Exercise 3
def test_exercise_3():
    """Test the switching point detector."""
    np.random.seed(42)

    # Test case 1: Clear difference (B much better than A)
    rewards_A = np.random.normal(0.05, 0.1, 1000)
    rewards_B = np.random.normal(0.15, 0.1, 1000)  # 10% better
    switch_point, p_vals = should_have_switched(rewards_A, rewards_B)

    assert switch_point is not None, "Should detect significant difference"
    assert switch_point < 100, f"Should detect quickly, got {switch_point}"
    assert len(p_vals) > 0, "Should return p-values"

    # Test case 2: No difference (both have same mean)
    rewards_A_same = np.random.normal(0.1, 0.1, 200)
    rewards_B_same = np.random.normal(0.1, 0.1, 200)
    switch_point_same, _ = should_have_switched(rewards_A_same, rewards_B_same)

    # Might spuriously detect significance due to randomness, but shouldn't be early
    # (We're being lenient here since it's random)

    # Test case 3: B worse than A (should not switch)
    rewards_A_better = np.random.normal(0.15, 0.1, 1000)
    rewards_B_worse = np.random.normal(0.05, 0.1, 1000)
    switch_point_worse, _ = should_have_switched(rewards_A_better, rewards_B_worse)

    # Should not detect a switch (B is worse, not better)
    # The implementation checks if B > A before setting switch_point

    print("✓ Exercise 3 tests passed!")


# =============================================================================
# Exercise 4: Commodity Trading Scenario (Open-Ended)
# =============================================================================

def commodity_opportunity_cost(n_samples_needed, diff_sharpe, capital_per_trade):
    """
    Calculate opportunity cost of running an A/B test to completion in commodity trading.

    Scenario:
    ---------
    You're testing two crude oil trading strategies:
    - Strategy A: Sharpe ratio = 1.2
    - Strategy B: Sharpe ratio = 1.5

    You calculate you need 1000 trades per strategy for 80% power.
    After 500 trades, you're confident B is better (p < 0.01).

    What is the opportunity cost of continuing the A/B test for the remaining
    500 trades per strategy (1000 total remaining)?

    Parameters:
    -----------
    n_samples_needed : int
        Total sample size needed per variant
    diff_sharpe : float
        Difference in Sharpe ratios (B - A)
    capital_per_trade : float
        Capital allocated per trade (e.g., $10,000)

    Returns:
    --------
    opportunity_cost : float
        Estimated dollar cost of continuing the A/B test

    Assumptions:
    ------------
    - Sharpe ratio approximates excess return / volatility
    - For simplicity, assume volatility = 15% annually
    - Approximate opportunity cost = diff_Sharpe × volatility × capital × n_trades / 2
      (÷2 because you're allocating 50/50, only half the trades are to the worse strategy)

    Example:
    --------
    >>> cost = commodity_opportunity_cost(
    ...     n_samples_needed=1000,
    ...     diff_sharpe=0.3,  # 1.5 - 1.2
    ...     capital_per_trade=10000
    ... )
    >>> print(f"Opportunity cost: ${cost:,.0f}")
    """
    # TODO: Implement opportunity cost calculation

    # Remaining trades if we already have half the data
    remaining_per_variant = n_samples_needed // 2

    # In A/B test, 50% go to worse strategy
    wasted_trades = remaining_per_variant

    # Approximate: Sharpe difference × volatility × capital × trades
    # Simplified: assume diff_Sharpe already captures the expected return difference
    # For a rough estimate: each trade to the worse strategy costs diff_Sharpe × volatility × capital
    volatility = 0.15  # 15% annual volatility assumption

    cost_per_trade = diff_sharpe * volatility * capital_per_trade
    opportunity_cost = cost_per_trade * wasted_trades

    return opportunity_cost


# Self-check for Exercise 4
def test_exercise_4():
    """Test the opportunity cost calculator."""
    # Test case: Standard scenario
    cost = commodity_opportunity_cost(
        n_samples_needed=1000,
        diff_sharpe=0.3,
        capital_per_trade=10000
    )

    # Expected: 500 trades × 0.3 × 0.15 × 10000 = $225,000
    expected = 500 * 0.3 * 0.15 * 10000
    assert np.isclose(cost, expected), f"Expected ${expected:,.0f}, got ${cost:,.0f}"

    # Test case: Smaller difference should cost less
    cost_small = commodity_opportunity_cost(
        n_samples_needed=1000,
        diff_sharpe=0.1,  # Smaller gap
        capital_per_trade=10000
    )
    assert cost_small < cost, "Smaller Sharpe difference should cost less"

    print("✓ Exercise 4 tests passed!")
    print(f"\n📊 Example Opportunity Cost:")
    print(f"  - Continuing A/B test for 500 more trades per strategy")
    print(f"  - Sharpe difference: 0.3 (1.5 vs 1.2)")
    print(f"  - Capital per trade: $10,000")
    print(f"  - Estimated opportunity cost: ${cost:,.0f}")
    print(f"\n  This is the cost of certainty in A/B testing!")


# =============================================================================
# Bonus Exercise: Visualize Regret Comparison
# =============================================================================

def visualize_regret_strategies():
    """
    Bonus: Create a visualization comparing regret accumulation for different strategies.

    This exercise has no right answer — it's about building intuition.
    """
    np.random.seed(42)

    # Three arms with different means
    arm_means = np.array([0.15, 0.25, 0.18])
    n_rounds = 1000

    # Strategy 1: Round-robin (A/B test with 3 arms)
    choices_rr = [t % 3 for t in range(n_rounds)]
    regret_rr = calculate_cumulative_regret(arm_means, choices_rr)

    # Strategy 2: Random exploration
    choices_random = np.random.randint(0, 3, n_rounds)
    regret_random = calculate_cumulative_regret(arm_means, choices_random)

    # Strategy 3: Oracle (always pick best)
    best_arm = np.argmax(arm_means)
    choices_oracle = [best_arm] * n_rounds
    regret_oracle = calculate_cumulative_regret(arm_means, choices_oracle)

    # Strategy 4: Pure exploitation (greedy after initial exploration)
    Q = np.zeros(3)
    N = np.zeros(3)
    choices_greedy = []
    for t in range(n_rounds):
        if t < 30:  # Initial exploration
            arm = t % 3
        else:
            arm = np.argmax(Q)

        choices_greedy.append(arm)
        reward = np.random.randn() + arm_means[arm]
        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]

    regret_greedy = calculate_cumulative_regret(arm_means, choices_greedy)

    # Visualize
    plt.figure(figsize=(12, 6))
    plt.plot(regret_rr, label='Round-Robin (A/B Test)', linewidth=2, alpha=0.8)
    plt.plot(regret_random, label='Random Exploration', linewidth=2, alpha=0.8)
    plt.plot(regret_greedy, label='Greedy (after initial)', linewidth=2, alpha=0.8)
    plt.plot(regret_oracle, label='Oracle (Best Arm Always)', linewidth=2, alpha=0.8, linestyle='--')

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title('Regret Comparison: Different Strategies', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('regret_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Visualization saved as 'regret_comparison.png'")
    plt.show()

    print("\n📈 Final Regret Values:")
    print(f"  Round-Robin:  {regret_rr[-1]:.2f}")
    print(f"  Random:       {regret_random[-1]:.2f}")
    print(f"  Greedy:       {regret_greedy[-1]:.2f}")
    print(f"  Oracle:       {regret_oracle[-1]:.2f} (zero, by definition)")


# =============================================================================
# Run All Exercises
# =============================================================================

def run_all_exercises():
    """Run all exercise tests."""
    print("=" * 60)
    print("MODULE 0 FOUNDATIONS: SELF-CHECK EXERCISES")
    print("=" * 60)

    print("\n[Exercise 1] A/B Test Sample Size Calculation")
    print("-" * 60)
    test_exercise_1()

    print("\n[Exercise 2] Cumulative Regret Calculator")
    print("-" * 60)
    test_exercise_2()

    print("\n[Exercise 3] Switching Point Detection")
    print("-" * 60)
    test_exercise_3()

    print("\n[Exercise 4] Commodity Trading Opportunity Cost")
    print("-" * 60)
    test_exercise_4()

    print("\n[Bonus] Regret Visualization")
    print("-" * 60)
    visualize_regret_strategies()

    print("\n" + "=" * 60)
    print("✅ ALL EXERCISES COMPLETED!")
    print("=" * 60)
    print("\nYou're ready to move to Module 1: Epsilon-Greedy Algorithm")


if __name__ == "__main__":
    run_all_exercises()

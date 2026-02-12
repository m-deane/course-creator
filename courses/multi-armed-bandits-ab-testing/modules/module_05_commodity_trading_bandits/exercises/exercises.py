"""
Module 5: Commodity Trading Bandits - Self-Check Exercises

These exercises test your understanding of:
- Reward function design
- Guardrail implementation
- Regime-aware allocation
- Two-wallet framework

All exercises are commodity-themed and include assert-based self-checks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================================================
# EXERCISE 1: Custom Reward Function
# ============================================================================

def exercise_1_custom_reward():
    """
    Design a reward function that combines Sharpe ratio with max drawdown penalty.

    Goal: Create a reward that balances returns, volatility, and drawdown protection.

    Requirements:
    - Incorporate mean return
    - Penalize volatility
    - Heavily penalize drawdowns > 10%
    """

    def custom_sharpe_drawdown_reward(
        returns: np.ndarray,
        lambda_dd: float = 2.0,
        dd_threshold: float = 0.10
    ) -> float:
        """
        Compute reward combining Sharpe ratio with drawdown penalty.

        Args:
            returns: Array of historical returns for one arm
            lambda_dd: Penalty weight for drawdowns
            dd_threshold: Drawdown threshold (10% = 0.10)

        Returns:
            Reward score (higher is better)
        """
        # YOUR CODE HERE
        # Hint 1: Compute mean return
        # Hint 2: Compute volatility (std)
        # Hint 3: Compute max drawdown from cumulative returns
        # Hint 4: Combine: sharpe - lambda_dd * max(0, drawdown - threshold)

        mean_return = returns.mean()
        volatility = returns.std()

        # Compute drawdown
        cumulative = (1 + returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1).min()

        # Sharpe-like ratio
        sharpe = mean_return / (volatility + 1e-6)

        # Drawdown penalty (only if exceeds threshold)
        dd_penalty = lambda_dd * max(0, abs(drawdown) - dd_threshold)

        return sharpe - dd_penalty

    # Test your implementation
    print("Exercise 1: Custom Reward Function")
    print("=" * 60)

    # Test case 1: Low drawdown (should have high reward)
    returns_low_dd = np.array([0.01, 0.02, 0.01, 0.015, 0.01])
    reward_low_dd = custom_sharpe_drawdown_reward(returns_low_dd)
    print(f"Low drawdown returns: {returns_low_dd}")
    print(f"Reward: {reward_low_dd:.4f}")

    # Test case 2: High drawdown (should have low reward)
    returns_high_dd = np.array([0.05, -0.20, 0.10, -0.15, 0.08])
    reward_high_dd = custom_sharpe_drawdown_reward(returns_high_dd)
    print(f"\nHigh drawdown returns: {returns_high_dd}")
    print(f"Reward: {reward_high_dd:.4f}")

    # Self-check
    assert reward_low_dd > reward_high_dd, "Low drawdown should have higher reward!"
    assert reward_high_dd < 0, "High drawdown should result in negative reward!"

    print("\n✓ Exercise 1 passed!")
    print("=" * 60)
    return custom_sharpe_drawdown_reward


# ============================================================================
# EXERCISE 2: Correlation Guardrail
# ============================================================================

def exercise_2_correlation_guardrail():
    """
    Implement a correlation guardrail that prevents concentration in correlated commodities.

    Goal: Ensure that highly correlated arms don't dominate the portfolio.

    Requirements:
    - Identify highly correlated pairs (correlation > 0.7)
    - Limit combined weight of correlated group to max_correlated_weight
    - Re-normalize to maintain valid allocation
    """

    def apply_correlation_guardrail(
        weights: np.ndarray,
        correlation_matrix: np.ndarray,
        max_correlated_weight: float = 0.50,
        corr_threshold: float = 0.7
    ) -> np.ndarray:
        """
        Apply correlation-based position limit.

        Args:
            weights: Proposed allocation weights (sums to 1)
            correlation_matrix: K x K correlation matrix
            max_correlated_weight: Max combined weight for correlated group
            corr_threshold: Correlation threshold to define "highly correlated"

        Returns:
            Adjusted weights (sums to 1)
        """
        # YOUR CODE HERE
        # Hint 1: Find pairs with correlation > corr_threshold
        # Hint 2: Group highly correlated arms
        # Hint 3: If group weight > max_correlated_weight, scale down proportionally
        # Hint 4: Re-normalize to sum to 1

        K = len(weights)
        adjusted = weights.copy()
        processed = set()

        for i in range(K):
            if i in processed:
                continue

            # Find correlated group for arm i
            correlated_group = [i]
            for j in range(K):
                if j != i and correlation_matrix[i, j] > corr_threshold:
                    correlated_group.append(j)

            # Check combined weight
            group_weight = adjusted[correlated_group].sum()

            if group_weight > max_correlated_weight:
                # Scale down proportionally
                scale = max_correlated_weight / group_weight
                adjusted[correlated_group] *= scale

            # Mark as processed
            processed.update(correlated_group)

        # Re-normalize
        return adjusted / adjusted.sum()

    # Test your implementation
    print("\nExercise 2: Correlation Guardrail")
    print("=" * 60)

    # Example: 5 commodities
    # WTI and Brent are highly correlated (0.95)
    # Gold and Silver are correlated (0.75)
    correlation_matrix = np.array([
        [1.00, 0.95, 0.20, 0.10, 0.15],  # WTI
        [0.95, 1.00, 0.15, 0.12, 0.18],  # Brent
        [0.20, 0.15, 1.00, 0.75, 0.10],  # Gold
        [0.10, 0.12, 0.75, 1.00, 0.08],  # Silver
        [0.15, 0.18, 0.10, 0.08, 1.00]   # Corn
    ])

    # Proposed weights: Heavy in WTI and Brent (correlated)
    proposed = np.array([0.40, 0.35, 0.15, 0.05, 0.05])
    print(f"Proposed weights: {proposed}")
    print(f"WTI + Brent (correlated): {proposed[0] + proposed[1]:.2%}")

    # Apply guardrail
    adjusted = apply_correlation_guardrail(proposed, correlation_matrix, max_correlated_weight=0.50)
    print(f"\nAdjusted weights: {adjusted}")
    print(f"WTI + Brent after guardrail: {adjusted[0] + adjusted[1]:.2%}")

    # Self-checks
    assert np.isclose(adjusted.sum(), 1.0), "Weights must sum to 1!"
    assert (adjusted[0] + adjusted[1]) <= 0.51, "Correlated pair should be capped!"
    assert all(adjusted >= 0), "No negative weights allowed!"

    print("\n✓ Exercise 2 passed!")
    print("=" * 60)
    return apply_correlation_guardrail


# ============================================================================
# EXERCISE 3: Seasonal-Aware Bandit
# ============================================================================

def exercise_3_seasonal_bandit():
    """
    Build a seasonal-aware bandit that adjusts beliefs based on commodity seasonality.

    Goal: Incorporate known seasonal patterns into the bandit's allocation.

    Requirements:
    - Define seasonal strength by month for each commodity
    - Adjust Thompson Sampling to favor commodities in their strong season
    - Maintain exploration even in weak seasons
    """

    class SeasonalBandit:
        """
        Thompson Sampling bandit with seasonal awareness.
        """

        def __init__(
            self,
            commodities: List[str],
            seasonal_patterns: Dict[str, Dict[int, float]],
            seasonal_boost: float = 0.5
        ):
            """
            Initialize seasonal-aware bandit.

            Args:
                commodities: List of commodity names
                seasonal_patterns: Dict mapping commodity -> month -> strength (-1 to 1)
                seasonal_boost: How much to boost allocation in strong seasons
            """
            self.commodities = commodities
            self.K = len(commodities)
            self.seasonal_patterns = seasonal_patterns
            self.seasonal_boost = seasonal_boost

            # Standard Thompson Sampling parameters
            self.means = np.zeros(self.K)
            self.stds = np.ones(self.K) * 0.02
            self.n = np.zeros(self.K)

        def get_seasonal_adjustment(self, month: int) -> np.ndarray:
            """
            Get seasonal adjustment factors for current month.

            Args:
                month: Current month (1-12)

            Returns:
                Array of adjustment factors (positive = boost, negative = reduce)
            """
            # YOUR CODE HERE
            # Hint: Look up seasonal strength for each commodity in current month
            # Hint: Return array of adjustments scaled by seasonal_boost

            adjustments = np.zeros(self.K)
            for i, commodity in enumerate(self.commodities):
                if commodity in self.seasonal_patterns:
                    seasonal_strength = self.seasonal_patterns[commodity].get(month, 0.0)
                    adjustments[i] = seasonal_strength * self.seasonal_boost

            return adjustments

        def get_allocation(self, month: int) -> np.ndarray:
            """
            Get allocation with seasonal adjustment.

            Args:
                month: Current month (1-12)

            Returns:
                Allocation weights (sums to 1)
            """
            # YOUR CODE HERE
            # Hint 1: Sample from each arm's posterior (Thompson Sampling)
            # Hint 2: Add seasonal adjustment to samples
            # Hint 3: Convert to weights via softmax

            # Thompson Sampling
            samples = np.random.normal(self.means, self.stds)

            # Add seasonal adjustment
            seasonal_adj = self.get_seasonal_adjustment(month)
            adjusted_samples = samples + seasonal_adj

            # Softmax to weights
            exp_samples = np.exp(adjusted_samples - adjusted_samples.max())
            weights = exp_samples / exp_samples.sum()

            return weights

        def update(self, returns: np.ndarray):
            """Standard Bayesian update."""
            for i in range(self.K):
                self.n[i] += 1
                lr = 1 / (self.n[i] + 1)
                self.means[i] = (1 - lr) * self.means[i] + lr * returns[i]
                self.stds[i] = self.stds[i] / np.sqrt(1 + self.n[i])

    # Test your implementation
    print("\nExercise 3: Seasonal-Aware Bandit")
    print("=" * 60)

    # Define seasonal patterns
    seasonal_patterns = {
        'Corn': {
            1: -0.5, 2: -0.3, 3: 0.0, 4: 0.3, 5: 0.5, 6: 0.8,
            7: 1.0, 8: 0.5, 9: 0.0, 10: -0.3, 11: -0.5, 12: -0.5
        },  # Strong in summer (growing season risk)
        'NatGas': {
            1: 1.0, 2: 0.8, 3: 0.3, 4: 0.0, 5: -0.3, 6: -0.5,
            7: -0.5, 8: -0.3, 9: 0.0, 10: 0.3, 11: 0.5, 12: 0.8
        },  # Strong in winter (heating demand)
        'WTI': {
            1: 0.0, 2: 0.0, 3: 0.3, 4: 0.5, 5: 0.8, 6: 1.0,
            7: 0.8, 8: 0.5, 9: 0.0, 10: -0.3, 11: -0.3, 12: 0.0
        }  # Strong in summer (driving season)
    }

    commodities = ['WTI', 'Gold', 'Copper', 'NatGas', 'Corn']
    bandit = SeasonalBandit(commodities, seasonal_patterns, seasonal_boost=0.3)

    # Test allocation in different months
    print("Testing seasonal allocation:")

    # January: NatGas should be favored (winter heating)
    weights_jan = bandit.get_allocation(month=1)
    print(f"\nJanuary allocation: {dict(zip(commodities, weights_jan))}")
    print(f"  NatGas (winter strong): {weights_jan[3]:.2%}")

    # July: Corn should be favored (growing season)
    weights_jul = bandit.get_allocation(month=7)
    print(f"\nJuly allocation: {dict(zip(commodities, weights_jul))}")
    print(f"  Corn (summer strong): {weights_jul[4]:.2%}")

    # Self-checks
    assert np.isclose(weights_jan.sum(), 1.0), "Weights must sum to 1!"
    assert np.isclose(weights_jul.sum(), 1.0), "Weights must sum to 1!"

    # NatGas should have higher allocation in January than July
    natgas_idx = 3
    assert weights_jan[natgas_idx] > weights_jul[natgas_idx], \
        "NatGas should be higher in winter (Jan) than summer (Jul)!"

    # Corn should have higher allocation in July than January
    corn_idx = 4
    assert weights_jul[corn_idx] > weights_jan[corn_idx], \
        "Corn should be higher in summer (Jul) than winter (Jan)!"

    print("\n✓ Exercise 3 passed!")
    print("=" * 60)
    return SeasonalBandit


# ============================================================================
# BONUS EXERCISE: Complete Two-Wallet System
# ============================================================================

def bonus_exercise_complete_system():
    """
    BONUS: Integrate all components into a complete two-wallet system.

    Components:
    - Custom reward function (Exercise 1)
    - Correlation guardrail (Exercise 2)
    - Seasonal awareness (Exercise 3)
    - Two-wallet framework

    This is a synthesis exercise - no implementation required,
    but think about how you'd combine all pieces.
    """

    print("\nBONUS Exercise: Complete System Design")
    print("=" * 60)
    print("\nThink about these questions:")
    print("1. In what order should you apply guardrails?")
    print("2. Should seasonal adjustment happen before or after Thompson Sampling?")
    print("3. How do you combine custom rewards with seasonal awareness?")
    print("4. What additional guardrails would you add for production?")
    print("\nAnswers:")
    print("1. Order: Volatility dampen → Position limits → Min allocation → Tilt speed")
    print("2. Seasonal adjustment AFTER sampling (boost the samples, not the beliefs)")
    print("3. Custom rewards inform the update step; seasonal adjusts the selection")
    print("4. Production adds: correlation limits, sector caps, transaction cost modeling")
    print("=" * 60)


# ============================================================================
# RUN ALL EXERCISES
# ============================================================================

def run_all_exercises():
    """Run all exercises with self-checks."""
    print("\n" + "=" * 60)
    print("MODULE 5: COMMODITY TRADING BANDITS - EXERCISES")
    print("=" * 60)

    try:
        # Exercise 1
        exercise_1_custom_reward()

        # Exercise 2
        exercise_2_correlation_guardrail()

        # Exercise 3
        exercise_3_seasonal_bandit()

        # Bonus
        bonus_exercise_complete_system()

        print("\n" + "=" * 60)
        print("ALL EXERCISES PASSED! 🎉")
        print("=" * 60)
        print("\nYou've mastered:")
        print("  ✓ Custom reward function design")
        print("  ✓ Correlation-based guardrails")
        print("  ✓ Seasonal-aware allocation")
        print("  ✓ Complete system integration")
        print("\nYou're ready to build production commodity allocators!")

    except AssertionError as e:
        print(f"\n❌ Exercise failed: {e}")
        print("Review the hints and try again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Check your implementation.")


if __name__ == "__main__":
    run_all_exercises()

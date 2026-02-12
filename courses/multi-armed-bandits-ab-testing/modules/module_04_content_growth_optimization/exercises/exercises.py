"""
Module 4 Exercises: Bandits for Content & Growth Optimization

Self-check exercises to practice designing and implementing
business bandit systems.

Run with: python exercises.py
"""

import numpy as np
from typing import List, Dict, Tuple


# ============================================================================
# EXERCISE 1: Design a Podcast Content Bandit
# ============================================================================

def exercise_1_design_podcast_bandit():
    """
    Design arms and rewards for a commodity trading podcast.

    Scenario: You publish 3 podcast episodes per week.

    Your task:
    1. Define 6 arms (2 formats × 3 topics)
    2. Choose an appropriate reward metric
    3. Specify exploration budget

    Fill in the dictionaries below.
    """

    # TODO: Define your arms
    # Example format: "Market Commentary × Solo Episode"
    arms = [
        # Add 6 arms here
        "Market Commentary × Solo",
        "Market Commentary × Interview",
        "Trading Strategy × Solo",
        "Trading Strategy × Interview",
        "Trader Stories × Solo",
        "Trader Stories × Interview"
    ]

    # TODO: Choose reward metric
    # Options: listen_through_rate, avg_listen_time, subscriber_growth,
    #          shares, ratings
    reward_metric = "listen_through_rate"  # Your choice

    # TODO: Set exploration budget (what % goes to non-top arms?)
    exploration_budget = 0.20  # 20% exploration

    # Self-check
    assert len(arms) == 6, "Should have 6 arms"
    assert 0 < exploration_budget < 0.5, "Exploration should be 5-40%"
    assert reward_metric in [
        "listen_through_rate", "avg_listen_time",
        "subscriber_growth", "shares", "ratings"
    ], "Invalid reward metric"

    print("Exercise 1: Podcast Bandit Design")
    print("=" * 60)
    print(f"Arms: {arms}")
    print(f"Reward metric: {reward_metric}")
    print(f"Exploration budget: {exploration_budget:.0%}")
    print("✓ Design looks good!\n")

    return arms, reward_metric, exploration_budget


# ============================================================================
# EXERCISE 2: Implement Minimum Traffic Constraint
# ============================================================================

class ConversionBanditWithMinimum:
    """
    Thompson Sampling bandit with minimum traffic constraint.

    Each arm must receive at least min_pulls before we can
    heavily exploit the best arm.
    """

    def __init__(self, n_variants: int, min_pulls_per_arm: int = 100):
        self.n = n_variants
        self.min_pulls = min_pulls_per_arm
        self.alpha = np.ones(n_variants)
        self.beta = np.ones(n_variants)
        self.pulls = np.zeros(n_variants)

    def select_variant(self) -> int:
        """
        Select variant using Thompson Sampling with minimum constraint.

        TODO: Implement the selection logic:
        1. Check if any arm has < min_pulls
        2. If yes, select from those arms (round-robin or random)
        3. If no, use standard Thompson Sampling
        """
        # Check for arms below minimum
        below_min = np.where(self.pulls < self.min_pulls)[0]

        if len(below_min) > 0:
            # TODO: Select from arms below minimum
            # Hint: Use np.random.choice(below_min)
            return np.random.choice(below_min)

        # All arms have minimum pulls, use Thompson Sampling
        # TODO: Sample from Beta posteriors and pick best
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, variant: int, converted: bool):
        """Update posterior after observing result"""
        self.pulls[variant] += 1
        if converted:
            self.alpha[variant] += 1
        else:
            self.beta[variant] += 1


def exercise_2_test_minimum_constraint():
    """
    Test the minimum traffic constraint bandit.
    """
    print("Exercise 2: Minimum Traffic Constraint")
    print("=" * 60)

    # Create bandit
    bandit = ConversionBanditWithMinimum(
        n_variants=4,
        min_pulls_per_arm=50
    )

    # Simulate with true conversion rates
    true_rates = [0.03, 0.06, 0.04, 0.05]

    # Run 500 visitors
    for _ in range(500):
        variant = bandit.select_variant()
        converted = np.random.rand() < true_rates[variant]
        bandit.update(variant, converted)

    # Check that all arms got minimum traffic
    print("Pulls per variant:")
    for i, pulls in enumerate(bandit.pulls):
        print(f"  Variant {i}: {int(pulls)} pulls "
              f"(min required: {bandit.min_pulls})")
        assert pulls >= bandit.min_pulls, \
            f"Variant {i} didn't reach minimum!"

    print("✓ All variants reached minimum traffic!\n")

    return bandit


# ============================================================================
# EXERCISE 3: Arm Retirement System
# ============================================================================

class BanditWithRetirement:
    """
    Bandit that can retire underperforming arms.
    """

    def __init__(self, initial_arms: List[str]):
        self.arms = initial_arms[:]
        self.n_pulls = {arm: 0 for arm in initial_arms}
        self.rewards = {arm: [] for arm in initial_arms}
        self.retired = []

    def get_mean(self, arm: str, window: int = None) -> float:
        """Get average reward (optionally windowed)"""
        if not self.rewards[arm]:
            return 0.0
        data = self.rewards[arm]
        if window:
            data = data[-window:]
        return np.mean(data)

    def should_retire(self, arm: str, min_pulls: int = 50,
                     threshold: float = 0.3) -> bool:
        """
        Determine if arm should be retired.

        TODO: Implement retirement logic:
        1. Has arm been pulled at least min_pulls times?
        2. Is arm the worst performer (by windowed mean)?
        3. Is windowed mean below absolute threshold?

        Return True only if ALL conditions are met.
        """
        # Check minimum pulls
        if self.n_pulls[arm] < min_pulls:
            return False

        # Get recent performance (last 20 pulls)
        recent_mean = self.get_mean(arm, window=20)

        # Check absolute threshold
        if recent_mean >= threshold:
            return False

        # Check if worst performer
        eligible_arms = [
            a for a in self.arms
            if self.n_pulls[a] >= min_pulls
        ]
        if len(eligible_arms) <= 2:
            return False  # Keep minimum 2 arms

        means = {a: self.get_mean(a, window=20)
                for a in eligible_arms}
        worst = min(eligible_arms, key=lambda a: means[a])

        return arm == worst

    def retire_arm(self, arm: str):
        """Retire an arm"""
        if arm in self.arms:
            self.arms.remove(arm)
            self.retired.append(arm)

    def add_arm(self, new_arm: str):
        """Add a new arm"""
        self.arms.append(new_arm)
        self.n_pulls[new_arm] = 0
        self.rewards[new_arm] = []

    def update(self, arm: str, reward: float):
        """Record observation"""
        self.n_pulls[arm] += 1
        self.rewards[arm].append(reward)


def exercise_3_test_retirement():
    """
    Test arm retirement system.
    """
    print("Exercise 3: Arm Retirement System")
    print("=" * 60)

    # Create bandit
    arms = ["Arm_A", "Arm_B", "Arm_C", "Arm_D"]
    bandit = BanditWithRetirement(arms)

    # Simulate different performance levels
    arm_performance = {
        "Arm_A": 0.50,  # Good
        "Arm_B": 0.45,  # Good
        "Arm_C": 0.38,  # Mediocre
        "Arm_D": 0.20   # Bad (should retire)
    }

    # Simulate 200 pulls (50 per arm to start)
    for _ in range(200):
        # Even exploration initially
        arm = np.random.choice(bandit.arms)
        reward = np.random.normal(
            arm_performance[arm], 0.1
        )
        reward = np.clip(reward, 0, 1)
        bandit.update(arm, reward)

    # Check for retirement
    print("Performance after 200 pulls:")
    for arm in arms:
        if arm in bandit.arms:
            mean = bandit.get_mean(arm, window=20)
            pulls = bandit.n_pulls[arm]
            should = bandit.should_retire(arm)
            print(f"  {arm}: {mean:.2f} avg, "
                  f"{pulls} pulls, retire={should}")

    # Arm_D should be eligible for retirement
    assert bandit.should_retire("Arm_D"), \
        "Arm_D should be eligible for retirement!"

    # Retire it
    bandit.retire_arm("Arm_D")
    print(f"\n✓ Retired: {bandit.retired}")
    print(f"✓ Active arms: {bandit.arms}\n")

    assert "Arm_D" not in bandit.arms
    assert "Arm_D" in bandit.retired

    return bandit


# ============================================================================
# BONUS EXERCISE: Design Your Own Business Bandit
# ============================================================================

def bonus_design_your_bandit():
    """
    Design a bandit for your own business problem.

    Fill in the template below with a real problem from your work.
    """

    design = {
        "problem": "Which commodity report format drives the most trades?",

        "arms": [
            "PDF with charts",
            "Email digest",
            "Video walkthrough",
            "Interactive dashboard"
        ],

        "reward_metric": "trades_made_within_24h",

        "context_features": [
            "market_volatility",  # High/low vol
            "trader_experience",  # New/veteran
            "asset_class"        # Energy/metals/ag
        ],

        "constraints": [
            "Must send at least one PDF per week (institutional clients)",
            "Video production limited to 2 per week"
        ],

        "algorithm": "Contextual Thompson Sampling",

        "exploration_budget": 0.15,  # 15%

        "evaluation_metric": "Lift in actionable trades vs baseline",

        "evaluation_period": "12 weeks"
    }

    print("Bonus: Design Your Own Business Bandit")
    print("=" * 60)
    for key, value in design.items():
        print(f"{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
    print()

    return design


# ============================================================================
# RUN ALL EXERCISES
# ============================================================================

def run_all_exercises():
    """Run all exercises with self-checks"""

    print("\n" + "=" * 60)
    print("MODULE 4 EXERCISES: Content & Growth Optimization")
    print("=" * 60 + "\n")

    # Exercise 1: Design
    arms, metric, budget = exercise_1_design_podcast_bandit()

    # Exercise 2: Minimum constraint
    bandit_min = exercise_2_test_minimum_constraint()

    # Exercise 3: Retirement
    bandit_retire = exercise_3_test_retirement()

    # Bonus: Your own design
    design = bonus_design_your_bandit()

    print("=" * 60)
    print("ALL EXERCISES COMPLETE! ✓")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Arms must be repeatable (not one-time events)")
    print("2. Reward metrics must align with business goals")
    print("3. Minimum traffic constraints ensure fair evaluation")
    print("4. Arm retirement prevents wasting traffic on dead options")
    print("5. Every business problem can be framed as a bandit")
    print("\nNext: Apply these patterns to your own work!")


if __name__ == "__main__":
    np.random.seed(42)
    run_all_exercises()

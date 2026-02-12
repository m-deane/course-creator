"""
Module 7: Production Systems - Self-Check Exercises

These exercises help you practice building production-ready bandit systems.
No grades, just learning. Run with: python exercises.py
"""

import numpy as np
from collections import defaultdict, deque, Counter


# ============================================================================
# EXERCISE 1: Cold Start Handler
# ============================================================================

def exercise_1_cold_start():
    """
    Exercise 1: Add a cold start handler to a bandit engine.

    Scenario: Your production bandit has 3 existing arms with data, but a new
    commodity (PLATINUM) just became available. How should the system handle
    this new arm with no historical data?

    Task: Implement a cold start strategy that gives new arms a fair chance
    without disrupting the existing policy.
    """

    class BanditWithColdStart:
        def __init__(self, epsilon=0.1, cold_start_rounds=20):
            self.epsilon = epsilon
            self.cold_start_rounds = cold_start_rounds
            self.alpha = defaultdict(lambda: 1.0)
            self.beta = defaultdict(lambda: 1.0)
            self.pull_counts = defaultdict(int)

        def add_arm(self, arm_id):
            """Add a new arm (cold start scenario)."""
            # TODO: Initialize new arm appropriately
            # Hint: Should you use optimistic initialization?
            pass

        def select_arm(self, arms):
            """Select arm with cold start handling."""
            # TODO: Implement cold start strategy
            # Options:
            # 1. Force-explore new arms for first N rounds
            # 2. Use optimistic initialization (high initial alpha)
            # 3. Increase epsilon temporarily when new arms present
            # 4. Dedicated exploration phase for new arms
            pass

        def update(self, arm, reward):
            """Update arm statistics."""
            self.alpha[arm] += reward
            self.beta[arm] += (1 - reward)
            self.pull_counts[arm] += 1

    # SOLUTION (uncomment to reveal)
    """
    class BanditWithColdStart:
        def __init__(self, epsilon=0.1, cold_start_rounds=20):
            self.epsilon = epsilon
            self.cold_start_rounds = cold_start_rounds
            self.alpha = defaultdict(lambda: 1.0)
            self.beta = defaultdict(lambda: 1.0)
            self.pull_counts = defaultdict(int)
            self.cold_start_arms = set()

        def add_arm(self, arm_id):
            # Optimistic initialization: assume new arm is good
            self.alpha[arm_id] = 10.0  # Optimistic
            self.beta[arm_id] = 5.0
            self.pull_counts[arm_id] = 0
            self.cold_start_arms.add(arm_id)

        def select_arm(self, arms):
            # Force-explore cold start arms for first N rounds
            for arm in self.cold_start_arms:
                if self.pull_counts[arm] < self.cold_start_rounds:
                    return arm

            # Remove from cold start set if explored enough
            self.cold_start_arms = {a for a in self.cold_start_arms
                                   if self.pull_counts[a] < self.cold_start_rounds}

            # Regular Thompson Sampling
            samples = {arm: np.random.beta(self.alpha[arm], self.beta[arm])
                      for arm in arms}
            return max(samples, key=samples.get)
    """

    # Test your implementation
    bandit = BanditWithColdStart()

    # Existing arms
    existing_arms = ["GOLD", "OIL", "NATGAS"]
    for arm in existing_arms:
        # Simulate some history
        bandit.update(arm, 0.5)
        bandit.update(arm, 0.6)

    # Add new arm
    bandit.add_arm("PLATINUM")

    # Check if new arm gets selected
    selections = []
    for _ in range(50):
        arm = bandit.select_arm(existing_arms + ["PLATINUM"])
        reward = np.random.random()
        bandit.update(arm, reward)
        selections.append(arm)

    platinum_pct = selections.count("PLATINUM") / len(selections)

    # Assert new arm gets reasonable exploration
    assert platinum_pct > 0.1, \
        f"Cold start arm PLATINUM only selected {platinum_pct:.1%} of the time!"

    print("✓ Exercise 1: Cold start handler works!")
    print(f"  PLATINUM selected {platinum_pct:.1%} of rounds (should be >10%)")


# ============================================================================
# EXERCISE 2: Offline Policy Evaluator
# ============================================================================

def exercise_2_offline_evaluation():
    """
    Exercise 2: Implement offline policy evaluation using IPS.

    Scenario: You have logged data from an epsilon-greedy policy and want to
    evaluate how a new greedy policy would have performed WITHOUT deploying it.

    Task: Implement inverse propensity scoring to estimate the new policy's
    expected reward.
    """

    # Logged data from epsilon-greedy policy (epsilon=0.2)
    logged_data = [
        {"context": {"vix": 20}, "arm": "GOLD", "reward": 0.02, "propensity": 0.85},
        {"context": {"vix": 25}, "arm": "GOLD", "reward": 0.01, "propensity": 0.82},
        {"context": {"vix": 18}, "arm": "OIL", "reward": 0.03, "propensity": 0.15},
        {"context": {"vix": 22}, "arm": "GOLD", "reward": 0.015, "propensity": 0.84},
        {"context": {"vix": 30}, "arm": "OIL", "reward": -0.01, "propensity": 0.18},
        {"context": {"vix": 19}, "arm": "GOLD", "reward": 0.025, "propensity": 0.86},
        {"context": {"vix": 27}, "arm": "OIL", "reward": 0.02, "propensity": 0.16},
    ]

    # New policy: always picks GOLD (greedy)
    def new_policy_probability(context, arm):
        """Return probability new policy assigns to arm given context."""
        # TODO: Implement new policy probabilities
        # This greedy policy always picks GOLD
        pass

    def ips_estimate(logged_data, new_policy_prob_fn):
        """Compute IPS estimate of new policy's expected reward."""
        # TODO: Implement IPS estimator
        # Formula: (1/n) * Σ [π₁(a|c) / π₀(a|c)] * r
        pass

    # SOLUTION (uncomment to reveal)
    """
    def new_policy_probability(context, arm):
        return 1.0 if arm == "GOLD" else 0.0

    def ips_estimate(logged_data, new_policy_prob_fn):
        total = 0.0
        for record in logged_data:
            context = record["context"]
            arm = record["arm"]
            reward = record["reward"]
            old_prob = record["propensity"]

            new_prob = new_policy_prob_fn(context, arm)
            weight = new_prob / old_prob if old_prob > 0 else 0
            total += weight * reward

        return total / len(logged_data)
    """

    # Test your implementation
    estimated_value = ips_estimate(logged_data, new_policy_probability)

    # Ground truth: GOLD's average reward in data
    gold_rewards = [d["reward"] for d in logged_data if d["arm"] == "GOLD"]
    true_value = np.mean(gold_rewards)

    print("✓ Exercise 2: Offline evaluation implemented!")
    print(f"  IPS estimate: {estimated_value:.4f}")
    print(f"  True GOLD average: {true_value:.4f}")

    # IPS should be close to true value (within sampling error)
    assert abs(estimated_value - true_value) < 0.02, \
        "IPS estimate too far from true value!"


# ============================================================================
# EXERCISE 3: Alert System
# ============================================================================

def exercise_3_alert_system():
    """
    Exercise 3: Build an alert system for performance degradation.

    Scenario: Your production bandit has been running for weeks. You need to
    detect when performance degrades below acceptable levels.

    Task: Implement an alert system that detects:
    1. Policy collapse (stuck on one arm)
    2. Reward degradation (performance below baseline)
    3. Excessive variance (unstable returns)
    """

    class BanditMonitor:
        def __init__(self, window_size=20, baseline_reward=0.5):
            self.window_size = window_size
            self.baseline_reward = baseline_reward
            self.recent_arms = deque(maxlen=window_size)
            self.recent_rewards = deque(maxlen=window_size)
            self.alerts = []

        def update(self, arm, reward):
            """Record a decision and its reward."""
            self.recent_arms.append(arm)
            self.recent_rewards.append(reward)

        def check_policy_collapse(self, entropy_threshold=0.5):
            """Alert if selection entropy too low (stuck on one arm)."""
            # TODO: Compute entropy of arm selections
            # If entropy < threshold, raise alert
            pass

        def check_reward_degradation(self, std_threshold=2.0):
            """Alert if recent rewards significantly below baseline."""
            # TODO: Check if recent_mean < baseline - (std_threshold * std)
            pass

        def check_excessive_variance(self, max_std=0.3):
            """Alert if reward variance too high (unstable)."""
            # TODO: Check if std(recent_rewards) > max_std
            pass

        def check_all_alerts(self):
            """Run all alert checks."""
            # TODO: Call all check methods, collect alerts
            pass

    # SOLUTION (uncomment to reveal)
    """
    class BanditMonitor:
        def __init__(self, window_size=20, baseline_reward=0.5):
            self.window_size = window_size
            self.baseline_reward = baseline_reward
            self.recent_arms = deque(maxlen=window_size)
            self.recent_rewards = deque(maxlen=window_size)
            self.alerts = []

        def update(self, arm, reward):
            self.recent_arms.append(arm)
            self.recent_rewards.append(reward)

        def check_policy_collapse(self, entropy_threshold=0.5):
            if len(self.recent_arms) < self.window_size:
                return False

            counts = Counter(self.recent_arms)
            probs = np.array([counts[a] for a in counts]) / len(self.recent_arms)
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            if entropy < entropy_threshold:
                alert = f"Policy collapse: entropy={entropy:.3f} < {entropy_threshold}"
                self.alerts.append(alert)
                return True
            return False

        def check_reward_degradation(self, std_threshold=2.0):
            if len(self.recent_rewards) < 10:
                return False

            mean_reward = np.mean(self.recent_rewards)
            std_reward = np.std(self.recent_rewards)

            if mean_reward < self.baseline_reward - std_threshold * std_reward:
                alert = f"Reward degradation: {mean_reward:.3f} << {self.baseline_reward:.3f}"
                self.alerts.append(alert)
                return True
            return False

        def check_excessive_variance(self, max_std=0.3):
            if len(self.recent_rewards) < 10:
                return False

            std = np.std(self.recent_rewards)
            if std > max_std:
                alert = f"Excessive variance: std={std:.3f} > {max_std}"
                self.alerts.append(alert)
                return True
            return False

        def check_all_alerts(self):
            self.alerts = []
            self.check_policy_collapse()
            self.check_reward_degradation()
            self.check_excessive_variance()
            return self.alerts
    """

    # Test Case 1: Policy collapse
    monitor1 = BanditMonitor(window_size=20)
    for _ in range(25):
        monitor1.update("GOLD", 0.5)  # Always GOLD
    alerts1 = monitor1.check_all_alerts()

    assert any("collapse" in a.lower() for a in alerts1), \
        "Should detect policy collapse when stuck on one arm!"

    # Test Case 2: Reward degradation
    monitor2 = BanditMonitor(window_size=20, baseline_reward=0.5)
    for _ in range(25):
        monitor2.update("ARM_A", 0.1)  # Poor performance
    alerts2 = monitor2.check_all_alerts()

    assert any("degradation" in a.lower() for a in alerts2), \
        "Should detect reward degradation!"

    # Test Case 3: Excessive variance
    monitor3 = BanditMonitor(window_size=20)
    for i in range(25):
        reward = 1.0 if i % 2 == 0 else 0.0  # Very volatile
        monitor3.update("ARM_B", reward)
    alerts3 = monitor3.check_all_alerts()

    assert any("variance" in a.lower() for a in alerts3), \
        "Should detect excessive variance!"

    print("✓ Exercise 3: Alert system working!")
    print(f"  Detected {len(alerts1)} alerts for policy collapse")
    print(f"  Detected {len(alerts2)} alerts for reward degradation")
    print(f"  Detected {len(alerts3)} alerts for excessive variance")


# ============================================================================
# RUN ALL EXERCISES
# ============================================================================

def run_all_exercises():
    """Run all exercises with error handling."""
    exercises = [
        ("Cold Start Handler", exercise_1_cold_start),
        ("Offline Policy Evaluator", exercise_2_offline_evaluation),
        ("Alert System", exercise_3_alert_system),
    ]

    print("=" * 70)
    print("MODULE 7: PRODUCTION SYSTEMS - EXERCISES")
    print("=" * 70)
    print()

    for name, exercise_fn in exercises:
        try:
            print(f"Running: {name}")
            print("-" * 70)
            exercise_fn()
            print()
        except AssertionError as e:
            print(f"❌ {name} failed: {e}")
            print("   Hint: Check the TODO comments and implement the missing logic\n")
        except Exception as e:
            print(f"❌ {name} error: {e}")
            print("   Debug your implementation\n")

    print("=" * 70)
    print("Exercises complete! Review any failures and try again.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_exercises()

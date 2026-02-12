"""
Module 1: Core Bandit Algorithms - Self-Check Exercises

Complete these exercises to test your understanding of epsilon-greedy, UCB1, and softmax.
Each exercise has assert statements for self-checking.

Run with: python exercises.py
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Exercise 1: Implement Epsilon-Greedy with Decaying Epsilon
# =============================================================================

def exercise_1():
    """
    Implement epsilon-greedy with decaying epsilon: ε(t) = 1/√(t+1)

    Requirements:
    - Start with ε=1 (pure exploration)
    - Decay to near-zero as t increases
    - Use the provided decay function
    """
    print("\n" + "="*70)
    print("EXERCISE 1: Decaying Epsilon-Greedy")
    print("="*70)

    class DecayingEpsilonGreedy:
        def __init__(self, k_arms):
            self.k = k_arms
            self.q_estimates = np.zeros(k_arms)
            self.action_counts = np.zeros(k_arms)
            self.t = 0

        def get_epsilon(self):
            """TODO: Implement epsilon decay schedule ε(t) = 1/√(t+1)"""
            # YOUR CODE HERE
            return 1.0 / np.sqrt(self.t + 1)

        def select_action(self):
            """Select action using decaying epsilon-greedy"""
            self.t += 1
            epsilon = self.get_epsilon()

            # YOUR CODE HERE: Implement epsilon-greedy selection
            if np.random.random() < epsilon:
                return np.random.randint(self.k)
            else:
                return np.argmax(self.q_estimates)

        def update(self, action, reward):
            """Update value estimates"""
            # YOUR CODE HERE: Implement incremental mean update
            self.action_counts[action] += 1
            n = self.action_counts[action]
            self.q_estimates[action] += (reward - self.q_estimates[action]) / n

    # Test the implementation
    agent = DecayingEpsilonGreedy(k_arms=5)

    # Create a simple bandit (arm 2 is best)
    true_means = np.array([0.2, 0.4, 0.8, 0.3, 0.1])

    for t in range(1000):
        action = agent.select_action()
        reward = np.random.normal(true_means[action], 0.1)
        reward = np.clip(reward, 0, 1)
        agent.update(action, reward)

    # Assertions
    assert agent.t == 1000, "Should have run 1000 steps"
    assert agent.get_epsilon() < 0.05, f"Epsilon should decay to <0.05, got {agent.get_epsilon():.3f}"
    assert np.argmax(agent.q_estimates) == 2, f"Should identify arm 2 as best, got {np.argmax(agent.q_estimates)}"
    assert agent.action_counts[2] > 500, f"Best arm should have >500 pulls, got {agent.action_counts[2]}"

    print("✓ Decaying epsilon implemented correctly")
    print(f"  Final epsilon: {agent.get_epsilon():.4f}")
    print(f"  Best arm identified: {np.argmax(agent.q_estimates)}")
    print(f"  Pull counts: {agent.action_counts.astype(int)}")
    print("\n✅ EXERCISE 1 PASSED!\n")


# =============================================================================
# Exercise 2: Modify UCB1 with Different Confidence Bound
# =============================================================================

def exercise_2():
    """
    Implement UCB1 with a different confidence bound formula.

    Instead of: √(ln(t)/N(a))
    Use: √(2·ln(t)/N(a))  [this is the standard UCB1]

    Then try: √(ln(t)/(N(a)+1))  [Laplace correction]
    """
    print("="*70)
    print("EXERCISE 2: UCB with Different Confidence Bounds")
    print("="*70)

    class ModifiedUCB:
        def __init__(self, k_arms, confidence_type='standard'):
            self.k = k_arms
            self.confidence_type = confidence_type
            self.q_estimates = np.zeros(k_arms)
            self.action_counts = np.zeros(k_arms)
            self.t = 0

        def get_confidence_bonus(self):
            """TODO: Implement different confidence bound formulas"""
            if self.confidence_type == 'standard':
                # YOUR CODE HERE: Standard UCB1 with √(2·ln(t)/N)
                return np.sqrt(2 * np.log(self.t) / (self.action_counts + 1e-10))

            elif self.confidence_type == 'laplace':
                # YOUR CODE HERE: Laplace correction √(ln(t)/(N+1))
                return np.sqrt(np.log(self.t) / (self.action_counts + 1))

            else:
                raise ValueError(f"Unknown confidence type: {self.confidence_type}")

        def select_action(self):
            self.t += 1

            # First K pulls: try each arm once
            if self.t <= self.k:
                return self.t - 1

            # YOUR CODE HERE: Compute UCB values and select best
            ucb_values = self.q_estimates + self.get_confidence_bonus()
            return np.argmax(ucb_values)

        def update(self, action, reward):
            # YOUR CODE HERE: Update estimates
            self.action_counts[action] += 1
            n = self.action_counts[action]
            self.q_estimates[action] += (reward - self.q_estimates[action]) / n

    # Test both variants
    true_means = np.array([0.3, 0.5, 0.7, 0.4, 0.2])

    for conf_type in ['standard', 'laplace']:
        agent = ModifiedUCB(k_arms=5, confidence_type=conf_type)

        for t in range(1000):
            action = agent.select_action()
            reward = np.random.normal(true_means[action], 0.2)
            reward = np.clip(reward, 0, 1)
            agent.update(action, reward)

        # Assertions
        assert np.argmax(agent.q_estimates) == 2, f"Should identify arm 2 as best"
        assert agent.action_counts[2] > 400, f"Best arm should have >400 pulls"

        print(f"\n✓ {conf_type.capitalize()} UCB implemented correctly")
        print(f"  Best arm: {np.argmax(agent.q_estimates)}")
        print(f"  Pull counts: {agent.action_counts.astype(int)}")

    print("\n✅ EXERCISE 2 PASSED!\n")


# =============================================================================
# Exercise 3: Run Algorithm Tournament on Custom Problem
# =============================================================================

def exercise_3():
    """
    Create a custom commodity trading scenario and compare all three algorithms.

    Scenario: 4 trading strategies with different risk-reward profiles
    - Strategy A: Low risk, low return
    - Strategy B: Medium risk, medium return
    - Strategy C: High risk, high return (best on average)
    - Strategy D: Very high risk, low return (worst)
    """
    print("="*70)
    print("EXERCISE 3: Algorithm Tournament on Custom Problem")
    print("="*70)

    class TradingBandit:
        """Bandit representing different trading strategies"""
        def __init__(self):
            # TODO: Define means and stds for 4 strategies
            # Make strategy C the best (highest mean)
            self.means = np.array([0.45, 0.55, 0.70, 0.30])  # YOUR CODE HERE
            self.stds = np.array([0.15, 0.25, 0.35, 0.50])   # YOUR CODE HERE
            self.k = 4
            self.best_arm = np.argmax(self.means)

        def pull(self, arm):
            reward = np.random.normal(self.means[arm], self.stds[arm])
            return np.clip(reward, 0, 1)

        def get_regret(self, arm):
            return np.max(self.means) - self.means[arm]

    # TODO: Implement three algorithms (copy from guides if needed)
    class EpsilonGreedy:
        def __init__(self, k):
            self.k = k
            self.q = np.zeros(k)
            self.n = np.zeros(k)

        def select(self, epsilon=0.1):
            if np.random.random() < epsilon:
                return np.random.randint(self.k)
            return np.argmax(self.q)

        def update(self, a, r):
            self.n[a] += 1
            self.q[a] += (r - self.q[a]) / self.n[a]

    class UCB1:
        def __init__(self, k):
            self.k = k
            self.q = np.zeros(k)
            self.n = np.zeros(k)
            self.t = 0

        def select(self):
            self.t += 1
            if self.t <= self.k:
                return self.t - 1
            ucb = self.q + np.sqrt(2 * np.log(self.t) / (self.n + 1e-10))
            return np.argmax(ucb)

        def update(self, a, r):
            self.n[a] += 1
            self.q[a] += (r - self.q[a]) / self.n[a]

    class Softmax:
        def __init__(self, k):
            self.k = k
            self.q = np.zeros(k)
            self.n = np.zeros(k)

        def select(self, tau=0.5):
            q_max = np.max(self.q)
            exp_q = np.exp((self.q - q_max) / tau)
            probs = exp_q / np.sum(exp_q)
            return np.random.choice(self.k, p=probs)

        def update(self, a, r):
            self.n[a] += 1
            self.q[a] += (r - self.q[a]) / self.n[a]

    # Run tournament
    np.random.seed(123)  # Set seed for reproducibility
    bandit = TradingBandit()
    T = 2000  # Longer horizon for more stable results

    algorithms = [
        ('ε-greedy', EpsilonGreedy(4)),
        ('UCB1', UCB1(4)),
        ('Softmax', Softmax(4))
    ]

    results = {}

    for name, algo in algorithms:
        total_regret = 0

        for t in range(T):
            if name == 'ε-greedy':
                action = algo.select(epsilon=0.1)
            elif name == 'UCB1':
                action = algo.select()
            else:  # Softmax
                action = algo.select(tau=0.5)

            reward = bandit.pull(action)
            algo.update(action, reward)
            total_regret += bandit.get_regret(action)

        results[name] = {
            'regret': total_regret,
            'best_arm_pct': 100 * algo.n[bandit.best_arm] / T
        }

    # Print results
    print("\nTournament Results:")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:12s}: Regret={res['regret']:6.2f}, Best Arm %={res['best_arm_pct']:5.1f}%")

    # Assertions
    # All algorithms should identify the best arm (strategy C)
    for name, res in results.items():
        assert res['best_arm_pct'] > 30, f"{name} should find best arm (got {res['best_arm_pct']:.1f}%)"

    # UCB1 should perform well (but allow for variance)
    assert results['UCB1']['best_arm_pct'] > 40, "UCB1 should focus on best arm"

    print("\n✅ EXERCISE 3 PASSED!\n")
    print("💡 Key insight: All algorithms eventually find the best arm!")


# =============================================================================
# Exercise 4: Implement Optimistic Initialization
# =============================================================================

def exercise_4():
    """
    Implement epsilon-greedy with optimistic initialization.

    Instead of Q(a) = 0, start with Q(a) = Q_init (e.g., 10).
    This encourages exploration early—every arm looks good initially!
    """
    print("="*70)
    print("EXERCISE 4: Optimistic Initialization")
    print("="*70)

    class OptimisticEpsilonGreedy:
        def __init__(self, k_arms, epsilon=0.1, q_init=5.0):
            self.k = k_arms
            self.epsilon = epsilon
            # TODO: Initialize Q with q_init instead of zeros
            self.q_estimates = np.full(k_arms, q_init)  # YOUR CODE HERE
            self.action_counts = np.zeros(k_arms)

        def select_action(self):
            if np.random.random() < self.epsilon:
                return np.random.randint(self.k)
            return np.argmax(self.q_estimates)

        def update(self, action, reward):
            self.action_counts[action] += 1
            n = self.action_counts[action]
            self.q_estimates[action] += (reward - self.q_estimates[action]) / n

    # Test optimistic vs standard initialization
    true_means = np.array([0.3, 0.5, 0.8, 0.4, 0.2])
    T = 500

    # Standard (pessimistic) initialization
    standard_agent = OptimisticEpsilonGreedy(k_arms=5, epsilon=0.0, q_init=0.0)

    # Optimistic initialization
    optimistic_agent = OptimisticEpsilonGreedy(k_arms=5, epsilon=0.0, q_init=5.0)

    for t in range(T):
        # Standard
        a1 = standard_agent.select_action()
        r1 = np.clip(np.random.normal(true_means[a1], 0.2), 0, 1)
        standard_agent.update(a1, r1)

        # Optimistic
        a2 = optimistic_agent.select_action()
        r2 = np.clip(np.random.normal(true_means[a2], 0.2), 0, 1)
        optimistic_agent.update(a2, r2)

    # Assertions
    # With ε=0 and Q_init=0, standard agent likely gets stuck on first arm
    # With Q_init=5, optimistic agent explores all arms before settling

    min_pulls_optimistic = np.min(optimistic_agent.action_counts)
    min_pulls_standard = np.min(standard_agent.action_counts)

    assert min_pulls_optimistic > min_pulls_standard, \
        "Optimistic init should explore all arms more evenly"

    print("\n✓ Optimistic initialization encourages exploration")
    print(f"\nStandard (Q_init=0, ε=0) pull counts: {standard_agent.action_counts.astype(int)}")
    print(f"  Min pulls: {min_pulls_standard:.0f} (some arms never tried!)")

    print(f"\nOptimistic (Q_init=5, ε=0) pull counts: {optimistic_agent.action_counts.astype(int)}")
    print(f"  Min pulls: {min_pulls_optimistic:.0f} (all arms explored!)")

    print("\n✅ EXERCISE 4 PASSED!\n")
    print("💡 Key insight: High Q_init → temporary optimism → exploration")


# =============================================================================
# Bonus Exercise: Implement UCB with Portfolio Constraints
# =============================================================================

def bonus_exercise():
    """
    BONUS: Implement a modified UCB for portfolio allocation with constraints.

    Constraint: Never allocate more than 50% to any single commodity.

    Modify UCB to respect this constraint while still exploring optimally.
    """
    print("="*70)
    print("BONUS EXERCISE: Constrained UCB for Portfolio Allocation")
    print("="*70)

    class ConstrainedUCB:
        def __init__(self, k_arms, max_allocation_pct=0.5):
            self.k = k_arms
            self.max_allocation = max_allocation_pct
            self.q_estimates = np.zeros(k_arms)
            self.action_counts = np.zeros(k_arms)
            self.t = 0

        def select_action(self):
            self.t += 1

            # First K pulls: try each once
            if self.t <= self.k:
                return self.t - 1

            # Compute UCB values
            ucb_values = self.q_estimates + np.sqrt(
                2 * np.log(self.t) / (self.action_counts + 1e-10)
            )

            # TODO: Apply constraint—if an arm has been pulled >50% of the time,
            # set its UCB to -inf so it won't be selected
            max_pulls = self.max_allocation * self.t

            # YOUR CODE HERE
            constrained_ucb = ucb_values.copy()
            for a in range(self.k):
                if self.action_counts[a] >= max_pulls:
                    constrained_ucb[a] = -np.inf

            return np.argmax(constrained_ucb)

        def update(self, action, reward):
            self.action_counts[action] += 1
            n = self.action_counts[action]
            self.q_estimates[action] += (reward - self.q_estimates[action]) / n

    # Test
    true_means = np.array([0.3, 0.5, 0.9, 0.4, 0.2])  # Arm 2 is much better
    agent = ConstrainedUCB(k_arms=5, max_allocation_pct=0.5)

    for t in range(1000):
        action = agent.select_action()
        reward = np.clip(np.random.normal(true_means[action], 0.2), 0, 1)
        agent.update(action, reward)

    # Assertions
    best_arm_pct = agent.action_counts[2] / agent.t
    assert best_arm_pct <= 0.51, f"Best arm allocation should be ≤50%, got {best_arm_pct:.2%}"
    assert best_arm_pct >= 0.45, f"Best arm should still get ~50%, got {best_arm_pct:.2%}"

    print("\n✓ Constraint respected!")
    print(f"  Best arm (2) allocation: {best_arm_pct:.1%}")
    print(f"  Pull distribution: {(100 * agent.action_counts / agent.t).astype(int)}%")

    print("\n✅ BONUS EXERCISE PASSED!\n")
    print("💡 Real-world application: Risk management constraints in portfolio allocation")


# =============================================================================
# Run all exercises
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" MODULE 1: CORE BANDIT ALGORITHMS - EXERCISES")
    print("="*70)

    try:
        exercise_1()
        exercise_2()
        exercise_3()
        exercise_4()

        print("\n" + "="*70)
        print(" ALL CORE EXERCISES PASSED! 🎉")
        print("="*70)

        # Bonus
        try:
            bonus_exercise()
            print("\n" + "="*70)
            print(" BONUS EXERCISE PASSED TOO! 🌟")
            print("="*70)
        except Exception as e:
            print(f"\nBonus exercise failed (optional): {e}")

        print("\n✅ You've mastered the core bandit algorithms!")
        print("   Next: Apply these to contextual bandits and A/B testing.")

    except AssertionError as e:
        print(f"\n❌ Exercise failed: {e}")
        print("   Review the code and try again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

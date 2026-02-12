"""
Self-Check Exercises: Contextual Bandits
Module 3: Multi-Armed Bandits & A/B Testing

Complete these exercises to test your understanding of contextual bandits,
LinUCB, and feature engineering for regime-aware allocation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# ============================================================================
# Exercise 1: Contextual Epsilon-Greedy
# ============================================================================

def exercise_1_contextual_epsilon_greedy():
    """
    Implement a contextual epsilon-greedy bandit that maintains separate
    reward estimates for different context buckets.

    Task: Complete the ContextualEpsilonGreedy class below.
    """

    class ContextualEpsilonGreedy:
        """Epsilon-greedy with context buckets."""

        def __init__(self, n_arms: int, n_context_buckets: int, epsilon: float = 0.1):
            self.n_arms = n_arms
            self.n_buckets = n_context_buckets
            self.epsilon = epsilon
            # Track rewards for each (bucket, arm) combination
            self.rewards = {}  # {(bucket, arm): [rewards]}

        def discretize_context(self, context: float) -> int:
            """Convert continuous context to bucket index."""
            # TODO: Implement context discretization
            # Hint: Map context value to bucket [0, n_buckets-1]
            # Example: if context is in [-1, 1], bucket 0 is [-1, -0.5), bucket 1 is [-0.5, 0), etc.
            bucket = int((context + 1) / 2 * self.n_buckets)
            return max(0, min(self.n_buckets - 1, bucket))

        def choose_arm(self, context: float) -> int:
            """Choose arm using epsilon-greedy within context bucket."""
            bucket = self.discretize_context(context)

            # TODO: Implement epsilon-greedy selection
            # With probability epsilon, choose random arm
            # Otherwise, choose arm with highest average reward in this bucket
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_arms)

            # Greedy: choose best arm for this bucket
            avg_rewards = []
            for arm in range(self.n_arms):
                key = (bucket, arm)
                if key in self.rewards and len(self.rewards[key]) > 0:
                    avg_rewards.append(np.mean(self.rewards[key]))
                else:
                    avg_rewards.append(0.0)

            return int(np.argmax(avg_rewards))

        def update(self, context: float, arm: int, reward: float):
            """Update reward history for (context bucket, arm)."""
            bucket = self.discretize_context(context)
            key = (bucket, arm)

            # TODO: Update reward history
            if key not in self.rewards:
                self.rewards[key] = []
            self.rewards[key].append(reward)

    # Test the implementation
    env = ContextualEpsilonGreedy(n_arms=3, n_context_buckets=5, epsilon=0.1)

    # Simulate some data
    for _ in range(100):
        context = np.random.uniform(-1, 1)
        arm = env.choose_arm(context)
        # Reward depends on context: arm 0 is best for negative context
        if context < 0:
            rewards = [0.8, 0.3, 0.4]
        else:
            rewards = [0.3, 0.7, 0.5]
        reward = rewards[arm] + np.random.normal(0, 0.1)
        env.update(context, arm, reward)

    # Test: In negative context, arm 0 should be chosen most
    negative_context = -0.5
    choices = [env.choose_arm(negative_context) for _ in range(100)]
    arm_0_freq = choices.count(0) / 100

    assert arm_0_freq > 0.5, f"Expected arm 0 to be chosen >50% in negative context, got {arm_0_freq:.1%}"
    print("✓ Exercise 1 passed: Contextual epsilon-greedy works!")
    return True


# ============================================================================
# Exercise 2: Feature Engineering and Evaluation
# ============================================================================

def exercise_2_feature_quality():
    """
    Add a new feature to a contextual bandit and measure improvement.

    Task: Implement feature extraction and evaluate feature importance.
    """

    def extract_features_basic(data: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features from price data."""
        features = pd.DataFrame(index=data.index)

        # Feature 1: Volatility
        returns = data.pct_change()
        vol = returns.rolling(20).std().mean(axis=1)
        features['volatility'] = (vol - vol.mean()) / vol.std()

        return features.fillna(0)

    def extract_features_enhanced(data: pd.DataFrame) -> pd.DataFrame:
        """Extract enhanced features including new one."""
        features = extract_features_basic(data)

        # TODO: Add a new feature that might improve performance
        # Ideas: trend (MA crossover), momentum, seasonality, correlation
        # Example: Add momentum feature
        returns = data.pct_change()
        momentum = returns.rolling(20).mean().mean(axis=1)
        features['momentum'] = (momentum - momentum.mean()) / momentum.std()

        return features.fillna(0)

    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    prices = pd.DataFrame({
        'asset_0': 100 * (1 + np.random.randn(500).cumsum() * 0.01),
        'asset_1': 100 * (1 + np.random.randn(500).cumsum() * 0.01),
        'asset_2': 100 * (1 + np.random.randn(500).cumsum() * 0.01)
    }, index=dates)

    # Extract features
    features_basic = extract_features_basic(prices)
    features_enhanced = extract_features_enhanced(prices)

    # Test: Enhanced features should have more columns
    assert features_enhanced.shape[1] > features_basic.shape[1], \
        "Enhanced features should have more columns than basic"

    # Test: All features should be normalized (mean ~0, std ~1)
    for col in features_enhanced.columns:
        mean = features_enhanced[col].mean()
        std = features_enhanced[col].std()
        assert abs(mean) < 0.2, f"Feature {col} not normalized (mean={mean:.2f})"
        assert 0.8 < std < 1.2, f"Feature {col} not normalized (std={std:.2f})"

    print(f"✓ Exercise 2 passed: Feature engineering works!")
    print(f"  Basic features: {features_basic.shape[1]}")
    print(f"  Enhanced features: {features_enhanced.shape[1]}")
    return True


# ============================================================================
# Exercise 3: Signal Routing with Contextual Bandits
# ============================================================================

def exercise_3_signal_routing():
    """
    Build a contextual bandit for routing trading signals to different models
    based on market regime.

    Task: Implement a signal router that chooses the best forecasting model
    for each regime.
    """

    class SignalRouter:
        """Route signals to best model based on market context."""

        def __init__(self, n_models: int, context_dim: int, alpha: float = 1.0):
            self.n_models = n_models
            self.d = context_dim
            self.alpha = alpha
            # LinUCB components
            self.A = [np.eye(context_dim) for _ in range(n_models)]
            self.b = [np.zeros(context_dim) for _ in range(n_models)]

        def choose_model(self, context: np.ndarray) -> int:
            """Choose best model for current context using LinUCB."""
            # TODO: Implement LinUCB model selection
            ucb_scores = []
            for m in range(self.n_models):
                theta = np.linalg.solve(self.A[m], self.b[m])
                pred = context @ theta
                std = np.sqrt(context @ np.linalg.solve(self.A[m], context))
                ucb = pred + self.alpha * std
                ucb_scores.append(ucb)
            return int(np.argmax(ucb_scores))

        def update(self, model: int, context: np.ndarray, accuracy: float):
            """Update model performance for given context."""
            # TODO: Update LinUCB statistics
            self.A[model] += np.outer(context, context)
            self.b[model] += accuracy * context

    # Simulate signal routing scenario
    router = SignalRouter(n_models=3, context_dim=2, alpha=1.0)

    # 3 models with different strengths:
    # Model 0: Good in low volatility (context[0] < 0)
    # Model 1: Good in high volatility (context[0] > 0)
    # Model 2: Always mediocre

    def get_model_accuracy(model: int, context: np.ndarray) -> float:
        """True accuracy of each model in each regime."""
        vol_regime = context[0]  # -1 to 1
        if model == 0:
            return 0.8 if vol_regime < 0 else 0.4
        elif model == 1:
            return 0.4 if vol_regime < 0 else 0.8
        else:
            return 0.5

    # Train the router
    for _ in range(200):
        context = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
        model = router.choose_model(context)
        accuracy = get_model_accuracy(model, context)
        accuracy += np.random.normal(0, 0.05)  # Add noise
        router.update(model, context, accuracy)

    # Test: Router should learn to use model 0 in low vol
    low_vol_context = np.array([-0.8, 0.0])
    choices = [router.choose_model(low_vol_context) for _ in range(50)]
    model_0_freq = choices.count(0) / 50

    assert model_0_freq > 0.6, \
        f"Expected model 0 chosen >60% in low vol, got {model_0_freq:.1%}"

    # Test: Router should learn to use model 1 in high vol
    high_vol_context = np.array([0.8, 0.0])
    choices = [router.choose_model(high_vol_context) for _ in range(50)]
    model_1_freq = choices.count(1) / 50

    assert model_1_freq > 0.6, \
        f"Expected model 1 chosen >60% in high vol, got {model_1_freq:.1%}"

    print("✓ Exercise 3 passed: Signal routing works!")
    print(f"  Low vol → Model 0: {model_0_freq:.1%}")
    print(f"  High vol → Model 1: {model_1_freq:.1%}")
    return True


# ============================================================================
# Exercise 4: Feature Importance Analysis
# ============================================================================

def exercise_4_feature_importance():
    """
    Analyze which features are most important for arm selection in LinUCB.

    Task: Implement ablation testing to measure feature importance.
    """

    class LinUCB:
        """Simple LinUCB for testing."""

        def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
            self.n_arms = n_arms
            self.d = context_dim
            self.alpha = alpha
            self.A = [np.eye(context_dim) for _ in range(n_arms)]
            self.b = [np.zeros(context_dim) for _ in range(n_arms)]

        def choose_arm(self, context: np.ndarray) -> int:
            ucb_scores = []
            for a in range(self.n_arms):
                theta = np.linalg.solve(self.A[a], self.b[a])
                pred = context @ theta
                std = np.sqrt(context @ np.linalg.solve(self.A[a], context))
                ucb_scores.append(pred + self.alpha * std)
            return int(np.argmax(ucb_scores))

        def update(self, arm: int, context: np.ndarray, reward: float):
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context

    def run_simulation(feature_mask: np.ndarray) -> float:
        """Run bandit simulation with selected features."""
        # True reward structure: r = 0.5*f1 + 0.3*f2 + 0.1*f3 for arm 0
        true_weights_full = np.array([
            [0.5, 0.3, 0.1],  # Arm 0
            [0.1, 0.5, 0.3],  # Arm 1
            [0.3, 0.1, 0.5]   # Arm 2
        ])

        # Mask weights to match selected features
        true_weights = true_weights_full[:, feature_mask]

        n_features = feature_mask.sum()
        bandit = LinUCB(n_arms=3, context_dim=n_features, alpha=1.0)

        cumulative_reward = 0
        for _ in range(300):
            # Generate context
            full_context = np.random.randn(3)
            context = full_context[feature_mask]

            # Choose arm
            arm = bandit.choose_arm(context)

            # Get reward (from masked context and weights)
            true_reward = context @ true_weights[arm]
            reward = true_reward + np.random.normal(0, 0.1)
            cumulative_reward += reward

            # Update
            bandit.update(arm, context, reward)

        return cumulative_reward

    # TODO: Test feature importance by ablation
    # Run with all features
    all_features = np.array([True, True, True])
    baseline_reward = run_simulation(all_features)

    # Run with each feature removed
    importance = {}
    for i in range(3):
        mask = all_features.copy()
        mask[i] = False
        reward_without_i = run_simulation(mask)
        importance[f'feature_{i}'] = baseline_reward - reward_without_i

    # Feature 0 should be most important (weight 0.5 vs 0.3, 0.1)
    most_important = max(importance, key=importance.get)

    print("✓ Exercise 4 passed: Feature importance analysis works!")
    print(f"  Feature importance: {importance}")
    print(f"  Most important: {most_important}")

    assert most_important == 'feature_0', \
        f"Expected feature_0 most important, got {most_important}"

    return True


# ============================================================================
# Run all exercises
# ============================================================================

def run_all_exercises():
    """Run all self-check exercises."""
    exercises = [
        ("Contextual Epsilon-Greedy", exercise_1_contextual_epsilon_greedy),
        ("Feature Engineering", exercise_2_feature_quality),
        ("Signal Routing", exercise_3_signal_routing),
        ("Feature Importance", exercise_4_feature_importance)
    ]

    print("=" * 60)
    print("Module 3: Contextual Bandits - Self-Check Exercises")
    print("=" * 60)

    results = []
    for name, exercise_func in exercises:
        print(f"\n[Running] {name}...")
        try:
            success = exercise_func()
            results.append((name, success))
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"✗ {name} error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} exercises passed")

    if passed == total:
        print("\n🎉 All exercises passed! You've mastered contextual bandits!")
    else:
        print("\n📚 Review the failed exercises and try again.")

    return passed == total


if __name__ == "__main__":
    run_all_exercises()

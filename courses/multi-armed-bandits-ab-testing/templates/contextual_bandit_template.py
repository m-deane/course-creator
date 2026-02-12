"""
Contextual Bandit with LinUCB - Copy and customize for your use case
Works with: Personalization, recommendation systems, context-dependent decisions
Time to working: 10 minutes

Use cases:
- Article recommendation (user features: age, location, interests)
- Ad placement (context features: time, page type, user segment)
- Treatment selection (patient features: age, biomarkers, medical history)
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Arms (options to choose from)
    "arms": ["option_a", "option_b", "option_c"],  # TODO: Customize arms

    # Feature engineering
    "feature_names": ["age", "location_score", "engagement_score"],  # TODO: Customize features
    "feature_dim": 3,  # Must match len(feature_names)

    # LinUCB parameters
    "alpha": 1.0,  # Exploration parameter (higher = more exploration)

    # Logging
    "log_level": "INFO",
}

# ============================================================================
# PRODUCTION-READY CONTEXTUAL BANDIT (COPY THIS ENTIRE BLOCK)
# ============================================================================

@dataclass
class LinUCBArm:
    """LinUCB arm with online learning"""
    name: str
    d: int  # Feature dimension

    def __post_init__(self):
        # A = D^T D (design matrix)
        self.A = np.identity(self.d)
        # b = D^T r (response vector)
        self.b = np.zeros(self.d)
        # Weight estimate: theta = A^-1 b
        self.theta = np.zeros(self.d)
        # Tracking
        self.pulls = 0

    def compute_ucb(self, x: np.ndarray, alpha: float) -> float:
        """Compute upper confidence bound for context x"""
        # UCB = theta^T x + alpha * sqrt(x^T A^-1 x)
        A_inv = np.linalg.inv(self.A)
        self.theta = A_inv @ self.b

        mean_reward = self.theta @ x
        uncertainty = alpha * np.sqrt(x.T @ A_inv @ x)

        return mean_reward + uncertainty

    def update(self, x: np.ndarray, reward: float):
        """Update parameters with observed (context, reward) pair"""
        self.A += np.outer(x, x)
        self.b += reward * x
        self.pulls += 1


class ContextualBanditEngine:
    """Contextual bandit with LinUCB algorithm"""

    def __init__(self, arms: List[str], feature_dim: int, alpha: float = 1.0, **kwargs):
        self.arms = {name: LinUCBArm(name=name, d=feature_dim) for name in arms}
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.config = kwargs
        self.total_pulls = 0
        self.history = []

        # Setup logging
        logging.basicConfig(
            level=kwargs.get("log_level", "INFO"),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def select_arm(self, context: Dict[str, float]) -> str:
        """Select arm based on context features using LinUCB"""
        # Convert context dict to feature vector
        x = self._featurize(context)

        # Compute UCB for each arm
        ucb_scores = {
            name: arm.compute_ucb(x, self.alpha)
            for name, arm in self.arms.items()
        }

        # Select arm with highest UCB
        selected_arm = max(ucb_scores.items(), key=lambda item: item[1])[0]

        self.logger.debug(f"Context: {context}, UCB scores: {ucb_scores}, Selected: {selected_arm}")

        return selected_arm

    def record_reward(self, arm: str, context: Dict[str, float], reward: float):
        """Update arm with observed reward"""
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")

        # Normalize reward to [0, 1]
        reward = np.clip(reward, 0.0, 1.0)

        # Convert context to features
        x = self._featurize(context)

        # Update arm
        self.arms[arm].update(x, reward)
        self.total_pulls += 1

        # Log
        self.history.append({
            "pull": self.total_pulls,
            "arm": arm,
            "context": context,
            "reward": reward
        })

        self.logger.info(f"Pull {self.total_pulls}: arm={arm}, reward={reward:.4f}, context={context}")

    def _featurize(self, context: Dict[str, float]) -> np.ndarray:
        """Convert context dict to feature vector"""
        # Extract features in consistent order
        feature_names = self.config.get("feature_names", [])

        if not feature_names:
            # Fallback: use dict values in sorted key order
            feature_names = sorted(context.keys())

        features = [context.get(name, 0.0) for name in feature_names]

        # Normalize features to [0, 1] range
        x = np.array(features, dtype=np.float64)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        return x

    def get_report(self) -> str:
        """Generate performance report"""
        lines = [
            "="*70,
            "Contextual Bandit Report (LinUCB)",
            "="*70,
            f"Total Pulls: {self.total_pulls}",
            f"Alpha (exploration): {self.alpha}",
            "",
            f"{'Arm':<20} {'Pulls':<10} {'Theta (weights)':<30}",
            "-"*70
        ]

        for name, arm in sorted(self.arms.items(), key=lambda x: x[1].pulls, reverse=True):
            theta_str = np.array2string(arm.theta, precision=3, suppress_small=True)
            lines.append(f"{name:<20} {arm.pulls:<10} {theta_str}")

        lines.append("="*70)
        return "\n".join(lines)


# ============================================================================
# RUN IT
# ============================================================================

def main():
    """Demonstration of contextual bandit"""

    # Initialize engine
    engine = ContextualBanditEngine(
        arms=CONFIG["arms"],
        feature_dim=CONFIG["feature_dim"],
        alpha=CONFIG["alpha"],
        feature_names=CONFIG["feature_names"],
        log_level=CONFIG["log_level"]
    )

    print(f"Running LinUCB contextual bandit")
    print(f"Arms: {CONFIG['arms']}")
    print(f"Features: {CONFIG['feature_names']}\n")

    # Simulate 200 rounds with varying contexts
    np.random.seed(42)

    for round_num in range(1, 201):
        # Generate random context
        context = {
            "age": np.random.uniform(18, 65),
            "location_score": np.random.uniform(0, 10),
            "engagement_score": np.random.uniform(0, 1)
        }

        # True reward model (unknown to bandit):
        # option_a: prefers young users with high engagement
        # option_b: prefers users with high location score
        # option_c: general option with baseline performance

        def true_reward(arm: str, ctx: Dict) -> float:
            if arm == "option_a":
                # Younger + engaged users
                score = (1 - ctx["age"] / 65) * 0.5 + ctx["engagement_score"] * 0.5
            elif arm == "option_b":
                # High location score
                score = ctx["location_score"] / 10
            elif arm == "option_c":
                # Baseline
                score = 0.4
            else:
                score = 0.0

            # Add noise
            return np.clip(score + np.random.normal(0, 0.1), 0, 1)

        # Select arm
        chosen_arm = engine.select_arm(context)

        # Simulate reward
        reward = true_reward(chosen_arm, context)

        # Record
        engine.record_reward(chosen_arm, context, reward)

    # Print final report
    print("\n" + engine.get_report())

    # Test learned policy
    print("\nTesting learned policy on specific contexts:")
    print("-" * 70)

    test_contexts = [
        {"age": 25, "location_score": 8, "engagement_score": 0.9},
        {"age": 50, "location_score": 3, "engagement_score": 0.3},
        {"age": 35, "location_score": 9, "engagement_score": 0.5},
    ]

    for ctx in test_contexts:
        selected = engine.select_arm(ctx)
        print(f"Context: {ctx}")
        print(f"  -> Selected arm: {selected}\n")


if __name__ == "__main__":
    main()

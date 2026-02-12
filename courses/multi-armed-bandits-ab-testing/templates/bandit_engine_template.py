"""
Multi-Armed Bandit Engine - Copy and customize for your use case
Works with: Any decision problem with discrete options and measurable outcomes
Time to working: 5 minutes

Example use cases:
- Content recommendation (which article/video to show)
- Ad placement (which ad variant performs best)
- Resource allocation (which server/region to route to)
- Product testing (which feature variant to deploy)
"""

import json
import logging
from typing import Dict, List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Arms (options) to choose from
    "arms": ["option_a", "option_b", "option_c"],

    # Policy: "epsilon_greedy", "ucb1", "thompson_sampling"
    "policy": "thompson_sampling",  # TODO: Customize policy

    # Policy parameters
    "epsilon": 0.1,  # For epsilon_greedy (0.1 = 10% exploration)
    "ucb_confidence": 2.0,  # For UCB1 (higher = more exploration)

    # Guardrails
    "min_pulls_per_arm": 10,  # Minimum samples before trusting estimates
    "max_allocation_pct": 0.8,  # Max 80% allocation to single arm

    # Logging
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
}

# ============================================================================
# PRODUCTION-READY BANDIT ENGINE (COPY THIS ENTIRE BLOCK)
# ============================================================================

@dataclass
class ArmStats:
    """Statistics for a single arm"""
    name: str
    pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    successes: int = 0  # For Thompson Sampling (Beta distribution)
    failures: int = 0   # For Thompson Sampling (Beta distribution)

    def update(self, reward: float):
        """Update statistics with new reward (0.0 to 1.0)"""
        self.pulls += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.pulls
        if reward >= 0.5:  # Binary success threshold
            self.successes += 1
        else:
            self.failures += 1


class BanditEngine:
    """Production-ready multi-armed bandit engine"""

    def __init__(self, arms: List[str], policy: str, **kwargs):
        self.arms = {name: ArmStats(name=name) for name in arms}
        self.policy = policy
        self.config = kwargs
        self.total_pulls = 0
        self.history = []

        # Setup logging
        logging.basicConfig(
            level=kwargs.get("log_level", "INFO"),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def select_arm(self, context: Optional[Dict] = None) -> str:
        """Select next arm to pull based on policy"""
        # Guardrail: ensure minimum pulls per arm
        min_pulls = self.config.get("min_pulls_per_arm", 10)
        unpulled = [name for name, stats in self.arms.items() if stats.pulls < min_pulls]
        if unpulled:
            arm = np.random.choice(unpulled)
            self.logger.debug(f"Min pulls guardrail: selecting {arm}")
            return arm

        # Select based on policy
        if self.policy == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.policy == "ucb1":
            return self._ucb1()
        elif self.policy == "thompson_sampling":
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

    def record_reward(self, arm: str, reward: float):
        """Record reward for chosen arm (normalized to 0.0-1.0)"""
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")

        # Normalize reward to [0, 1] if needed
        reward = np.clip(reward, 0.0, 1.0)

        self.arms[arm].update(reward)
        self.total_pulls += 1
        self.history.append({
            "pull": self.total_pulls,
            "arm": arm,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })

        self.logger.info(f"Pull {self.total_pulls}: arm={arm}, reward={reward:.4f}")

    def get_stats(self) -> Dict:
        """Get current statistics for all arms"""
        return {name: asdict(stats) for name, stats in self.arms.items()}

    def get_report(self) -> str:
        """Get human-readable performance report"""
        lines = [
            "="*60,
            f"Bandit Engine Report (Policy: {self.policy})",
            "="*60,
            f"Total Pulls: {self.total_pulls}\n",
            f"{'Arm':<15} {'Pulls':<8} {'Mean Reward':<12} {'Win Rate':<10}",
            "-"*60
        ]

        for name, stats in sorted(self.arms.items(), key=lambda x: x[1].mean_reward, reverse=True):
            win_rate = stats.successes / stats.pulls if stats.pulls > 0 else 0.0
            lines.append(
                f"{name:<15} {stats.pulls:<8} {stats.mean_reward:<12.4f} {win_rate:<10.2%}"
            )

        lines.append("="*60)
        return "\n".join(lines)

    # Policy implementations
    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy: explore with probability epsilon, else exploit"""
        epsilon = self.config.get("epsilon", 0.1)
        if np.random.random() < epsilon:
            # Explore: random arm
            return np.random.choice(list(self.arms.keys()))
        else:
            # Exploit: best arm
            return max(self.arms.items(), key=lambda x: x[1].mean_reward)[0]

    def _ucb1(self) -> str:
        """Upper Confidence Bound: balance mean reward + uncertainty"""
        confidence = self.config.get("ucb_confidence", 2.0)

        def ucb_score(stats: ArmStats) -> float:
            if stats.pulls == 0:
                return float('inf')
            exploitation = stats.mean_reward
            exploration = np.sqrt((confidence * np.log(self.total_pulls)) / stats.pulls)
            return exploitation + exploration

        return max(self.arms.items(), key=lambda x: ucb_score(x[1]))[0]

    def _thompson_sampling(self) -> str:
        """Thompson Sampling: sample from posterior Beta distributions"""
        samples = {}
        for name, stats in self.arms.items():
            # Beta(successes + 1, failures + 1) posterior
            alpha = stats.successes + 1
            beta = stats.failures + 1
            samples[name] = np.random.beta(alpha, beta)

        return max(samples.items(), key=lambda x: x[1])[0]


# ============================================================================
# RUN IT
# ============================================================================

def main():
    """Demonstration of bandit engine usage"""

    # Initialize engine
    engine = BanditEngine(
        arms=CONFIG["arms"],
        policy=CONFIG["policy"],
        epsilon=CONFIG["epsilon"],
        ucb_confidence=CONFIG["ucb_confidence"],
        min_pulls_per_arm=CONFIG["min_pulls_per_arm"],
        log_level=CONFIG["log_level"]
    )

    # Simulate decision-making loop
    print(f"Running {CONFIG['policy']} bandit with arms: {CONFIG['arms']}\n")

    # True reward probabilities (unknown to bandit)
    true_rewards = {
        "option_a": 0.3,
        "option_b": 0.5,
        "option_c": 0.7  # Best option
    }

    # Run 100 rounds
    for round_num in range(1, 101):
        # Select arm
        chosen_arm = engine.select_arm()

        # Simulate reward (Bernoulli with true probability)
        reward = 1.0 if np.random.random() < true_rewards[chosen_arm] else 0.0

        # Record result
        engine.record_reward(chosen_arm, reward)

    # Print final report
    print("\n" + engine.get_report())

    # Export statistics
    stats = engine.get_stats()
    with open("bandit_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nStatistics exported to bandit_stats.json")


if __name__ == "__main__":
    main()

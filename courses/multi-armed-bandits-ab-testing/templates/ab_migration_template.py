"""
A/B Test to Bandit Migration Template - Copy and customize for your use case
Works with: Gradual migration from A/B testing to adaptive bandits
Time to working: 5 minutes

Strategy: Start with A/B test (burn-in), then switch to bandit once statistically significant
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Literal
from scipy import stats
from datetime import datetime
import logging

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Arms to test
    "arms": ["variant_a", "variant_b", "variant_c"],  # TODO: Customize variants

    # Burn-in phase (A/B testing)
    "burn_in_rounds": 100,  # TODO: Customize burn-in period
    "min_samples_per_arm": 30,  # Minimum samples before statistical test
    "significance_threshold": 0.05,  # p-value threshold for switching

    # Bandit phase
    "bandit_policy": "thompson_sampling",  # epsilon_greedy, ucb1, thompson_sampling
    "epsilon": 0.1,  # For epsilon_greedy

    # Logging
    "log_level": "INFO",
}

# ============================================================================
# PRODUCTION-READY MIGRATION SYSTEM (COPY THIS ENTIRE BLOCK)
# ============================================================================

class ABtoBanditMigrator:
    """Migrate from A/B testing to bandit optimization"""

    def __init__(self, arms: List[str], config: Dict):
        self.arms = arms
        self.config = config

        # Statistics tracking
        self.stats = {
            arm: {"pulls": 0, "successes": 0, "failures": 0, "rewards": []}
            for arm in arms
        }

        # Phase tracking
        self.current_phase = "burn_in"  # "burn_in" or "bandit"
        self.total_pulls = 0
        self.switch_round = None

        # History
        self.history = []

        # Setup logging
        logging.basicConfig(
            level=config.get("log_level", "INFO"),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def select_arm(self) -> str:
        """Select arm based on current phase"""
        # Check if we should switch to bandit phase
        if self.current_phase == "burn_in":
            if self._should_switch_to_bandit():
                self.current_phase = "bandit"
                self.switch_round = self.total_pulls
                self.logger.info(f"SWITCHING TO BANDIT at round {self.total_pulls}")

        # Select arm based on phase
        if self.current_phase == "burn_in":
            return self._ab_select()
        else:
            return self._bandit_select()

    def record_reward(self, arm: str, reward: float):
        """Record reward for chosen arm"""
        if arm not in self.stats:
            raise ValueError(f"Unknown arm: {arm}")

        # Normalize reward to [0, 1]
        reward = np.clip(reward, 0.0, 1.0)

        # Update statistics
        self.stats[arm]["pulls"] += 1
        self.stats[arm]["rewards"].append(reward)
        if reward >= 0.5:
            self.stats[arm]["successes"] += 1
        else:
            self.stats[arm]["failures"] += 1

        self.total_pulls += 1

        # Log
        self.history.append({
            "round": self.total_pulls,
            "phase": self.current_phase,
            "arm": arm,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })

        self.logger.debug(f"Round {self.total_pulls} [{self.current_phase}]: {arm} -> {reward:.4f}")

    def _ab_select(self) -> str:
        """A/B testing: uniform random selection"""
        return np.random.choice(self.arms)

    def _bandit_select(self) -> str:
        """Bandit selection based on configured policy"""
        policy = self.config["bandit_policy"]

        if policy == "thompson_sampling":
            samples = {}
            for arm, arm_stats in self.stats.items():
                alpha = arm_stats["successes"] + 1
                beta = arm_stats["failures"] + 1
                samples[arm] = np.random.beta(alpha, beta)
            return max(samples.items(), key=lambda x: x[1])[0]

        elif policy == "epsilon_greedy":
            epsilon = self.config["epsilon"]
            if np.random.random() < epsilon:
                return np.random.choice(self.arms)
            else:
                return max(
                    self.stats.items(),
                    key=lambda x: x[1]["successes"] / max(x[1]["pulls"], 1)
                )[0]

        elif policy == "ucb1":
            def ucb_score(arm_stats):
                if arm_stats["pulls"] == 0:
                    return float('inf')
                mean = arm_stats["successes"] / arm_stats["pulls"]
                exploration = np.sqrt(2 * np.log(self.total_pulls) / arm_stats["pulls"])
                return mean + exploration

            return max(self.stats.items(), key=lambda x: ucb_score(x[1]))[0]

        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _should_switch_to_bandit(self) -> bool:
        """Determine if we should switch from A/B to bandit"""
        # Not enough rounds yet
        if self.total_pulls < self.config["burn_in_rounds"]:
            return False

        # Not enough samples per arm
        min_samples = self.config["min_samples_per_arm"]
        if any(arm_stats["pulls"] < min_samples for arm_stats in self.stats.values()):
            return False

        # Check for statistical significance
        return self._check_significance()

    def _check_significance(self) -> bool:
        """Check if there's a statistically significant difference between arms"""
        # Use chi-squared test for proportions
        observed = np.array([
            [self.stats[arm]["successes"], self.stats[arm]["failures"]]
            for arm in self.arms
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(observed)

        self.logger.info(f"Significance test: chi2={chi2:.4f}, p={p_value:.4f}")

        # Switch if p-value below threshold (significant difference exists)
        return p_value < self.config["significance_threshold"]

    def get_report(self) -> str:
        """Generate performance report"""
        lines = [
            "="*70,
            "A/B to Bandit Migration Report",
            "="*70,
            f"Current Phase: {self.current_phase.upper()}",
            f"Total Rounds: {self.total_pulls}",
        ]

        if self.switch_round is not None:
            lines.append(f"Switched to Bandit at Round: {self.switch_round}")

        lines.extend([
            "",
            f"{'Arm':<20} {'Pulls':<10} {'Success Rate':<15} {'Mean Reward':<12}",
            "-"*70
        ])

        for arm, arm_stats in sorted(
            self.stats.items(),
            key=lambda x: x[1]["successes"] / max(x[1]["pulls"], 1),
            reverse=True
        ):
            success_rate = arm_stats["successes"] / max(arm_stats["pulls"], 1)
            mean_reward = np.mean(arm_stats["rewards"]) if arm_stats["rewards"] else 0.0
            lines.append(
                f"{arm:<20} {arm_stats['pulls']:<10} {success_rate:<15.2%} {mean_reward:<12.4f}"
            )

        lines.append("="*70)
        return "\n".join(lines)


# ============================================================================
# RUN IT
# ============================================================================

def main():
    """Demonstration of A/B to bandit migration"""

    # Initialize migrator
    migrator = ABtoBanditMigrator(
        arms=CONFIG["arms"],
        config=CONFIG
    )

    # True reward probabilities (unknown to system)
    true_rewards = {
        "variant_a": 0.3,
        "variant_b": 0.5,
        "variant_c": 0.7  # Best variant
    }

    print(f"Starting migration experiment with {CONFIG['arms']}")
    print(f"Burn-in: {CONFIG['burn_in_rounds']} rounds")
    print(f"Bandit policy: {CONFIG['bandit_policy']}\n")

    # Run 300 rounds
    for round_num in range(1, 301):
        # Select arm
        chosen_arm = migrator.select_arm()

        # Simulate reward
        reward = 1.0 if np.random.random() < true_rewards[chosen_arm] else 0.0

        # Record
        migrator.record_reward(chosen_arm, reward)

        # Print phase switch
        if round_num == migrator.switch_round:
            print(f"\n>>> SWITCHED TO BANDIT PHASE at round {round_num} <<<\n")

    # Final report
    print("\n" + migrator.get_report())

    # Export history
    history_df = pd.DataFrame(migrator.history)
    history_df.to_csv("migration_history.csv", index=False)
    print("\nHistory exported to migration_history.csv")


if __name__ == "__main__":
    main()

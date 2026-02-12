"""
Evaluation and Monitoring Patterns - Track bandit performance
Each recipe solves ONE evaluation problem in < 20 lines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ============================================================================
# RECIPE 1: Cumulative Regret Plot (15 lines)
# Problem: Visualize opportunity cost of exploration
# ============================================================================

def plot_cumulative_regret(history: List[Dict], best_arm_reward: float,
                           save_path: str = "regret.png"):
    """Plot cumulative regret over time

    Args:
        history: [{"arm": str, "reward": float}, ...]
        best_arm_reward: True reward of optimal arm (if known)
        save_path: Path to save plot
    """
    regrets = [best_arm_reward - h["reward"] for h in history]
    cumulative_regret = np.cumsum(regrets)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_regret, linewidth=2)
    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title("Bandit Performance: Lower is Better", fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Regret plot saved to {save_path}")

# Example usage:
# history = [{"arm": "A", "reward": 0.7}, {"arm": "B", "reward": 0.5}, {"arm": "A", "reward": 0.8}]
# plot_cumulative_regret(history, best_arm_reward=0.8, save_path="regret.png")


# ============================================================================
# RECIPE 2: Arm Selection Distribution (14 lines)
# Problem: Visualize exploration patterns (are we exploring enough?)
# ============================================================================

def plot_arm_distribution(history: List[Dict], save_path: str = "arm_dist.png"):
    """Plot arm selection frequency over time

    Args:
        history: [{"arm": str, "reward": float}, ...]
        save_path: Path to save plot
    """
    df = pd.DataFrame(history)
    arm_counts = df["arm"].value_counts()

    plt.figure(figsize=(8, 6))
    arm_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.xlabel("Arm", fontsize=12)
    plt.ylabel("Number of Pulls", fontsize=12)
    plt.title("Exploration Pattern", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to {save_path}")

# Example usage:
# history = [{"arm": "A", "reward": 0.7}, {"arm": "B", "reward": 0.5}, {"arm": "A", "reward": 0.8}]
# plot_arm_distribution(history, save_path="arm_dist.png")


# ============================================================================
# RECIPE 3: Policy Comparison (20 lines)
# Problem: Compare multiple bandit algorithms on same data
# ============================================================================

def compare_policies(policies: Dict[str, callable], true_rewards: Dict[str, float],
                    num_rounds: int = 1000) -> pd.DataFrame:
    """Run multiple policies and compare cumulative rewards

    Args:
        policies: {"policy_name": selection_function, ...}
        true_rewards: {"arm": true_mean_reward, ...}
        num_rounds: Number of rounds to simulate
    Returns:
        DataFrame with cumulative rewards per policy
    """
    results = {name: [] for name in policies}

    for policy_name, select_fn in policies.items():
        cumulative_reward = 0
        for _ in range(num_rounds):
            arm = select_fn()  # Policy selects arm
            reward = 1.0 if np.random.random() < true_rewards[arm] else 0.0
            cumulative_reward += reward
            results[policy_name].append(cumulative_reward)

    df = pd.DataFrame(results)
    return df

# Example usage:
# policies = {"random": lambda: np.random.choice(["A", "B"]), "greedy": lambda: "A"}
# true_rewards = {"A": 0.7, "B": 0.5}
# comparison = compare_policies(policies, true_rewards, num_rounds=100)


# ============================================================================
# RECIPE 4: Offline Policy Evaluation (IPS) (18 lines)
# Problem: Evaluate new policy using logged data (without live testing)
# ============================================================================

def inverse_propensity_score(logged_data: pd.DataFrame, new_policy_probs: Dict[str, float]) -> float:
    """Evaluate new policy using Inverse Propensity Scoring

    Args:
        logged_data: DataFrame with ["arm", "reward", "logging_prob"] columns
        new_policy_probs: {"arm": selection_probability, ...} for new policy
    Returns:
        Estimated mean reward of new policy
    """
    ips_rewards = []

    for _, row in logged_data.iterrows():
        arm = row["arm"]
        reward = row["reward"]
        logging_prob = row["logging_prob"]  # Probability old policy selected this arm
        new_prob = new_policy_probs.get(arm, 0)

        # IPS estimator: reweight by probability ratio
        if logging_prob > 0:
            ips_reward = (new_prob / logging_prob) * reward
            ips_rewards.append(ips_reward)

    return np.mean(ips_rewards) if ips_rewards else 0.0

# Example usage:
# logged_data = pd.DataFrame({"arm": ["A", "B", "A"], "reward": [1, 0, 1], "logging_prob": [0.5, 0.5, 0.5]})
# new_policy_probs = {"A": 0.8, "B": 0.2}  # New policy prefers A
# estimated_reward = inverse_propensity_score(logged_data, new_policy_probs)


# ============================================================================
# RECIPE 5: Anomaly Detection (16 lines)
# Problem: Detect when bandit's behavior degrades (e.g., reward distribution shifts)
# ============================================================================

def detect_reward_anomaly(recent_rewards: List[float], historical_rewards: List[float],
                         threshold_std: float = 2.0) -> Tuple[bool, float]:
    """Detect if recent rewards are anomalously different from history

    Args:
        recent_rewards: Recent observed rewards (e.g., last 50)
        historical_rewards: Historical baseline rewards
        threshold_std: Number of standard deviations for anomaly threshold
    Returns:
        (is_anomaly: bool, z_score: float)
    """
    if len(historical_rewards) < 10:
        return False, 0.0  # Not enough data

    historical_mean = np.mean(historical_rewards)
    historical_std = np.std(historical_rewards)
    recent_mean = np.mean(recent_rewards)

    # Z-score: how many standard deviations is recent mean from historical?
    z_score = (recent_mean - historical_mean) / (historical_std + 1e-8)
    is_anomaly = abs(z_score) > threshold_std

    return is_anomaly, z_score

# Example usage:
# historical = [0.5, 0.6, 0.55, 0.58, 0.52, 0.6, 0.57, 0.59]
# recent = [0.3, 0.35, 0.32, 0.28]  # Anomalously low
# is_anomaly, z_score = detect_reward_anomaly(recent, historical, threshold_std=2.0)


# ============================================================================
# RECIPE 6: Confidence Interval for Arm Estimates (12 lines)
# Problem: Show uncertainty in arm reward estimates
# ============================================================================

def compute_reward_confidence_interval(arm_rewards: List[float],
                                      confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute confidence interval for arm's true reward

    Args:
        arm_rewards: List of observed rewards for this arm
        confidence: Confidence level (e.g., 0.95 = 95%)
    Returns:
        (mean, lower_bound, upper_bound)
    """
    mean_reward = np.mean(arm_rewards)
    std_error = np.std(arm_rewards) / np.sqrt(len(arm_rewards))
    z_critical = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    margin = z_critical * std_error
    return mean_reward, mean_reward - margin, mean_reward + margin

# Example usage:
# arm_rewards = [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.65]
# mean, lower, upper = compute_reward_confidence_interval(arm_rewards, confidence=0.95)


# ============================================================================
# RECIPE 7: Statistical Power Calculation (14 lines)
# Problem: How many samples needed to detect difference between arms?
# ============================================================================

def sample_size_for_power(effect_size: float, power: float = 0.8,
                         alpha: float = 0.05) -> int:
    """Calculate required sample size per arm for statistical power

    Args:
        effect_size: Minimum detectable difference (e.g., 0.1 = 10% difference)
        power: Desired statistical power (e.g., 0.8 = 80%)
        alpha: Significance level (e.g., 0.05 = 5%)
    Returns:
        Required samples per arm
    """
    # Simplified formula (assumes equal variance)
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return int(np.ceil(n))

# Example usage:
# samples_needed = sample_size_for_power(effect_size=0.1, power=0.8, alpha=0.05)


# ============================================================================
# RECIPE 8: Thompson Sampling Credible Intervals (13 lines)
# Problem: Visualize uncertainty in Thompson Sampling posteriors
# ============================================================================

def thompson_credible_intervals(arms: Dict[str, Dict],
                               credible_mass: float = 0.95) -> Dict[str, Tuple]:
    """Compute credible intervals for Thompson Sampling arms

    Args:
        arms: {"arm": {"successes": int, "failures": int}, ...}
        credible_mass: Credible interval mass (e.g., 0.95 = 95%)
    Returns:
        {"arm": (median, lower, upper), ...}
    """
    from scipy.stats import beta

    intervals = {}
    for arm, stats in arms.items():
        alpha, beta_param = stats["successes"] + 1, stats["failures"] + 1
        lower = beta.ppf((1 - credible_mass) / 2, alpha, beta_param)
        upper = beta.ppf(1 - (1 - credible_mass) / 2, alpha, beta_param)
        median = beta.ppf(0.5, alpha, beta_param)
        intervals[arm] = (median, lower, upper)

    return intervals

# Example usage:
# arms = {"A": {"successes": 20, "failures": 10}, "B": {"successes": 15, "failures": 15}}
# intervals = thompson_credible_intervals(arms, credible_mass=0.95)

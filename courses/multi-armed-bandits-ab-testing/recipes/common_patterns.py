"""
Common Bandit Patterns - Copy-paste code snippets for quick implementation
Each pattern solves ONE specific problem in < 20 lines
"""

import numpy as np
from typing import Dict, List

# ============================================================================
# PATTERN 1: Quick Thompson Sampling (10 lines)
# Problem: Need simplest possible Thompson Sampling implementation
# ============================================================================

def thompson_sampling_select(arms: Dict[str, Dict]) -> str:
    """Select arm using Thompson Sampling (Beta-Bernoulli)

    Args:
        arms: {"arm_name": {"successes": int, "failures": int}, ...}
    Returns:
        Selected arm name
    """
    samples = {
        name: np.random.beta(stats["successes"] + 1, stats["failures"] + 1)
        for name, stats in arms.items()
    }
    return max(samples.items(), key=lambda x: x[1])[0]

# Example usage:
# arms = {"A": {"successes": 10, "failures": 5}, "B": {"successes": 8, "failures": 3}}
# chosen = thompson_sampling_select(arms)


# ============================================================================
# PATTERN 2: Epsilon-Greedy with Decay (15 lines)
# Problem: Want exploration to decrease over time
# ============================================================================

def epsilon_greedy_decay(arms: Dict[str, float], round_num: int,
                         initial_epsilon: float = 0.3, decay_rate: float = 0.995) -> str:
    """Epsilon-greedy with exponential decay

    Args:
        arms: {"arm_name": mean_reward, ...}
        round_num: Current round number (1-indexed)
        initial_epsilon: Starting exploration rate
        decay_rate: Decay multiplier per round (0.995 = ~99.5% per round)
    Returns:
        Selected arm name
    """
    epsilon = initial_epsilon * (decay_rate ** round_num)

    if np.random.random() < epsilon:
        return np.random.choice(list(arms.keys()))  # Explore
    else:
        return max(arms.items(), key=lambda x: x[1])[0]  # Exploit

# Example usage:
# arms = {"A": 0.65, "B": 0.45, "C": 0.52}
# chosen = epsilon_greedy_decay(arms, round_num=50, initial_epsilon=0.3)


# ============================================================================
# PATTERN 3: UCB1 with Custom Confidence (12 lines)
# Problem: Need UCB with adjustable exploration strength
# ============================================================================

def ucb1_custom(arms: Dict[str, Dict], total_pulls: int, confidence: float = 2.0) -> str:
    """UCB1 with adjustable confidence parameter

    Args:
        arms: {"arm_name": {"pulls": int, "mean_reward": float}, ...}
        total_pulls: Total number of pulls across all arms
        confidence: Exploration strength (higher = more exploration)
    Returns:
        Selected arm name
    """
    def ucb_score(stats: Dict) -> float:
        if stats["pulls"] == 0:
            return float('inf')
        exploitation = stats["mean_reward"]
        exploration = np.sqrt((confidence * np.log(total_pulls)) / stats["pulls"])
        return exploitation + exploration

    return max(arms.items(), key=lambda x: ucb_score(x[1]))[0]

# Example usage:
# arms = {"A": {"pulls": 20, "mean_reward": 0.6}, "B": {"pulls": 15, "mean_reward": 0.7}}
# chosen = ucb1_custom(arms, total_pulls=35, confidence=2.0)


# ============================================================================
# PATTERN 4: Arm Retirement (18 lines)
# Problem: Drop underperforming arms after enough evidence
# ============================================================================

def retire_poor_arms(arms: Dict[str, Dict], min_pulls: int = 50,
                     threshold_pct: float = 0.5) -> List[str]:
    """Retire arms performing below threshold

    Args:
        arms: {"arm_name": {"pulls": int, "mean_reward": float}, ...}
        min_pulls: Minimum pulls before considering retirement
        threshold_pct: Retire if < threshold_pct * best_arm_reward
    Returns:
        List of active arm names
    """
    # Find best performer
    eligible = {k: v for k, v in arms.items() if v["pulls"] >= min_pulls}
    if not eligible:
        return list(arms.keys())  # Keep all if not enough data

    best_reward = max(v["mean_reward"] for v in eligible.values())
    retirement_threshold = threshold_pct * best_reward

    # Keep arms above threshold or with insufficient data
    active = [
        name for name, stats in arms.items()
        if stats["pulls"] < min_pulls or stats["mean_reward"] >= retirement_threshold
    ]
    return active

# Example usage:
# arms = {"A": {"pulls": 100, "mean_reward": 0.7}, "B": {"pulls": 100, "mean_reward": 0.3}}
# active_arms = retire_poor_arms(arms, min_pulls=50, threshold_pct=0.6)


# ============================================================================
# PATTERN 5: Non-Stationary Bandit with Sliding Window (15 lines)
# Problem: Environment changes over time, need recent data emphasis
# ============================================================================

def sliding_window_reward(rewards: List[float], window_size: int = 50) -> float:
    """Calculate mean reward using sliding window (for non-stationary environments)

    Args:
        rewards: List of all historical rewards for this arm
        window_size: Number of recent rewards to consider
    Returns:
        Mean reward over recent window
    """
    if len(rewards) == 0:
        return 0.0

    recent_rewards = rewards[-window_size:]
    return np.mean(recent_rewards)

# Example usage:
# all_rewards = [0.5, 0.6, 0.4, 0.7, 0.8, 0.9, 0.85, 0.95]  # Recent rewards higher
# recent_mean = sliding_window_reward(all_rewards, window_size=4)  # Uses last 4 only


# ============================================================================
# PATTERN 6: Reward Normalization (16 lines)
# Problem: Different arms have different reward scales
# ============================================================================

def normalize_rewards(arms: Dict[str, List[float]], method: str = "minmax") -> Dict[str, float]:
    """Normalize rewards across arms to [0, 1] scale

    Args:
        arms: {"arm_name": [reward1, reward2, ...], ...}
        method: "minmax" or "zscore"
    Returns:
        {"arm_name": normalized_mean_reward, ...}
    """
    all_rewards = [r for rewards in arms.values() for r in rewards]

    if method == "minmax":
        min_r, max_r = min(all_rewards), max(all_rewards)
        return {
            name: (np.mean(rewards) - min_r) / (max_r - min_r + 1e-8)
            for name, rewards in arms.items()
        }
    elif method == "zscore":
        mean_r, std_r = np.mean(all_rewards), np.std(all_rewards)
        return {
            name: (np.mean(rewards) - mean_r) / (std_r + 1e-8)
            for name, rewards in arms.items()
        }

# Example usage:
# arms = {"A": [10, 12, 11], "B": [100, 110, 105]}  # Different scales
# normalized = normalize_rewards(arms, method="minmax")


# ============================================================================
# PATTERN 7: Softmax Exploration (12 lines)
# Problem: Want probabilistic selection weighted by performance
# ============================================================================

def softmax_select(arms: Dict[str, float], temperature: float = 0.1) -> str:
    """Select arm using softmax (Boltzmann) exploration

    Args:
        arms: {"arm_name": mean_reward, ...}
        temperature: Lower = more exploitation, higher = more exploration
    Returns:
        Selected arm name
    """
    exp_rewards = {name: np.exp(reward / temperature) for name, reward in arms.items()}
    total = sum(exp_rewards.values())
    probabilities = {name: exp_r / total for name, exp_r in exp_rewards.items()}

    return np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))

# Example usage:
# arms = {"A": 0.7, "B": 0.5, "C": 0.4}
# chosen = softmax_select(arms, temperature=0.1)


# ============================================================================
# PATTERN 8: Optimistic Initialization (10 lines)
# Problem: Encourage initial exploration of all arms
# ============================================================================

def optimistic_init(arms: List[str], initial_value: float = 1.0) -> Dict[str, Dict]:
    """Initialize arms with optimistic reward estimates

    Args:
        arms: List of arm names
        initial_value: Optimistic initial mean reward (> true expected reward)
    Returns:
        {"arm_name": {"pulls": int, "mean_reward": float}, ...}
    """
    return {name: {"pulls": 1, "mean_reward": initial_value} for name in arms}

# Example usage:
# arms_dict = optimistic_init(["A", "B", "C"], initial_value=0.9)

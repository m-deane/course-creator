"""
Recipe: GRPO Advantage Computation
===================================

Copy-paste recipe for computing group-relative advantages.
This is the core of GRPO — normalize rewards within a group
so only relative ordering drives learning.

Usage:
    rewards = [0.3, 0.5, 0.7, 0.9]
    advantages = compute_grpo_advantages(rewards)
"""

import numpy as np


def compute_grpo_advantages(rewards: list[float]) -> list[float]:
    """
    Normalize rewards within a group to get advantages.

    Positive advantage = better than group average (reinforce)
    Negative advantage = worse than group average (suppress)

    >>> compute_grpo_advantages([0.3, 0.5, 0.7, 0.9])
    [-1.34, -0.45, 0.45, 1.34]  # approximately
    """
    r = np.array(rewards, dtype=np.float64)
    mean, std = r.mean(), r.std()

    if std < 1e-8:
        return [0.0] * len(rewards)

    return ((r - mean) / std).tolist()


def grpo_clipped_loss(
    log_probs: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    clip_epsilon: float = 0.2,
) -> float:
    """
    Compute GRPO clipped surrogate loss.

    Parameters
    ----------
    log_probs : array
        Log probabilities under current policy
    old_log_probs : array
        Log probabilities under old policy
    advantages : array
        Normalized advantages from compute_grpo_advantages
    clip_epsilon : float
        Clipping range (default 0.2)

    Returns
    -------
    float - loss value (to be minimized)
    """
    ratio = np.exp(log_probs - old_log_probs)
    clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    loss = -np.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    return float(loss)


if __name__ == "__main__":
    # Example: 4 completions for "What is the capital of France?"
    rewards = [0.3, 0.5, 0.7, 0.9]
    advantages = compute_grpo_advantages(rewards)

    print("GRPO Advantage Computation")
    print("=" * 50)
    print(f"Rewards:    {rewards}")
    print(f"Mean:       {np.mean(rewards):.2f}")
    print(f"Std:        {np.std(rewards):.2f}")
    print(f"Advantages: {[f'{a:+.2f}' for a in advantages]}")
    print()

    for r, a in zip(rewards, advantages):
        action = "REINFORCE" if a > 0 else "SUPPRESS "
        bar = "+" * int(abs(a) * 10) if a > 0 else "-" * int(abs(a) * 10)
        print(f"  Reward {r:.1f} -> Advantage {a:+.2f} [{action}] {bar}")

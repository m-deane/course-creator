"""
Module 05 — Training Loop Deep-Dive
Self-Check Exercise: Implement and Instrument a Simplified Training Loop

This exercise implements a simplified RL training loop using only NumPy and
standard library. No ART, vLLM, or LLM API calls are needed — the "model"
and "reward function" are synthetic, which lets you focus on the training
loop mechanics without infrastructure dependencies.

You will implement:
  1. Advantage normalization (the GRPO group scoring step)
  2. A policy update rule (simplified — log-probability weighted update)
  3. Checkpoint save and load utilities
  4. A training loop that tracks reward improvement over steps
  5. A reward trend plot to verify learning is happening

Run with:
    python 01_training_loop_exercise.py

All self-checks use assert statements. The script prints PASSED or a
diagnostic message for each exercise.
"""

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# =============================================================================
# Synthetic environment: the "agent" and "reward function"
#
# The synthetic agent has a hidden correct action (0 or 1) for each scenario.
# The policy is a probability distribution over {0, 1}.
# The reward is 1.0 if the agent picks the correct action, 0.0 otherwise,
# plus a small amount of noise to simulate RULER-style soft rewards.
# =============================================================================

@dataclass
class Scenario:
    """A training scenario with a known correct action."""
    scenario_id: str
    correct_action: int       # 0 or 1
    difficulty: float         # 0.0 = easy (noise-free), 1.0 = hard (very noisy)


@dataclass
class Trajectory:
    """One rollout from the synthetic agent."""
    scenario_id: str
    action: int               # Which action the agent took
    reward: Optional[float] = None  # Set after scoring


class SyntheticPolicy:
    """
    A simple policy represented as a probability p(action=1) for each scenario.

    At initialization, p=0.5 (random). After training, p should move toward
    p=1.0 if action=1 is correct, or p=0.0 if action=0 is correct.

    This stands in for the LoRA-modified language model. The "weights" here
    are just one float per scenario. In real training, they are millions of
    LoRA parameters.
    """

    def __init__(self, scenarios: list[Scenario], initial_p: float = 0.5):
        self.probs: dict[str, float] = {s.scenario_id: initial_p for s in scenarios}

    def sample_action(self, scenario_id: str) -> int:
        """Sample an action from the current policy distribution."""
        p = self.probs[scenario_id]
        return 1 if random.random() < p else 0

    def update(self, scenario_id: str, action: int, advantage: float, learning_rate: float = 0.05):
        """
        Simplified policy gradient update.

        Adjusts p(action=1) based on which action was taken and whether it
        was above or below the group average.

        - action=1 with positive advantage → increase p(action=1)
        - action=1 with negative advantage → decrease p(action=1)
        - action=0 with positive advantage → decrease p(action=1)
        - action=0 with negative advantage → increase p(action=1)

        This mirrors the GRPO policy gradient: reinforce trajectories that
        were above-average within their group, suppress those below-average.
        The direction of update depends on what the agent actually did.
        """
        current_p = self.probs[scenario_id]
        logit = math.log(current_p / (1 - current_p + 1e-8) + 1e-8)
        # Gradient of log π(action|state) w.r.t. logit:
        #   action=1: grad = (1 - p)   → positive update increases p
        #   action=0: grad = -p        → positive update decreases p
        if action == 1:
            grad = 1 - current_p
        else:
            grad = -current_p
        logit += learning_rate * advantage * grad * 4  # scale factor for toy model
        new_p = 1 / (1 + math.exp(-logit))
        # Clip to avoid degenerate distributions
        self.probs[scenario_id] = max(0.01, min(0.99, new_p))

    def save(self, path: str):
        """Save policy weights (the probability dictionary) to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "policy_weights.json", "w") as f:
            json.dump(self.probs, f, indent=2)

    @classmethod
    def load(cls, scenarios: list[Scenario], path: str) -> "SyntheticPolicy":
        """Load policy weights from a previously saved checkpoint."""
        policy = cls(scenarios)
        weights_path = Path(path) / "policy_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"No weights found at {weights_path}")
        with open(weights_path) as f:
            policy.probs = json.load(f)
        return policy


def synthetic_reward(scenario: Scenario, action: int) -> float:
    """
    Score an action for a scenario.

    Returns 1.0 if action is correct, 0.0 if wrong, with Gaussian noise
    scaled by difficulty. This simulates RULER's soft scoring.
    """
    base_reward = 1.0 if action == scenario.correct_action else 0.0
    noise = random.gauss(0, scenario.difficulty * 0.15)
    return float(max(0.0, min(1.0, base_reward + noise)))


# =============================================================================
# Exercise 1: Advantage Normalization
#
# Given a group of rewards (from N rollouts on the same scenario),
# compute the group-normalized advantage for each trajectory.
#
# Formula: A_i = (r_i - mean(group)) / (std(group) + epsilon)
#
# If std is near zero (all rewards identical), all advantages should be 0.
# =============================================================================

def compute_advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    """
    Compute group-normalized advantages from a list of rewards.

    Parameters
    ----------
    rewards : list[float]
        Reward for each trajectory in the group.
    epsilon : float
        Small value added to std to prevent division by zero.

    Returns
    -------
    list[float]
        Normalized advantage for each trajectory. Same length as rewards.
        Advantages sum to approximately 0 (before floating point error).

    Examples
    --------
    >>> compute_advantages([0.9, 0.7, 0.5, 0.3])
    [1.341..., 0.447..., -0.447..., -1.341...]

    >>> compute_advantages([0.8, 0.8, 0.8])
    [0.0, 0.0, 0.0]
    """
    # --- Implement this function ---
    n = len(rewards)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance)
    return [(r - mean) / (std + epsilon) for r in rewards]


def test_exercise_1():
    """Self-check: advantage normalization."""
    # Test 1: Basic four-element group
    rewards = [0.9, 0.7, 0.5, 0.3]
    advantages = compute_advantages(rewards)
    assert len(advantages) == 4, f"Expected 4 advantages, got {len(advantages)}"
    assert abs(sum(advantages)) < 1e-6, (
        f"Advantages should sum to ~0, got {sum(advantages):.6f}"
    )
    assert advantages[0] > advantages[1] > advantages[2] > advantages[3], (
        "Advantages should be in descending order (highest reward → highest advantage)"
    )
    assert advantages[0] > 0, "Best trajectory should have positive advantage"
    assert advantages[-1] < 0, "Worst trajectory should have negative advantage"

    # Test 2: Uniform rewards → all-zero advantages (no learning signal)
    uniform_rewards = [0.8, 0.8, 0.8, 0.8]
    uniform_advantages = compute_advantages(uniform_rewards)
    for a in uniform_advantages:
        assert abs(a) < 1e-4, (
            f"Uniform rewards should produce near-zero advantages, got {a:.6f}"
        )

    # Test 3: Two-element group
    two_rewards = [1.0, 0.0]
    two_advantages = compute_advantages(two_rewards)
    assert two_advantages[0] > 0 and two_advantages[1] < 0, (
        "Higher reward should have positive advantage"
    )
    assert abs(two_advantages[0] + two_advantages[1]) < 1e-6, (
        "Two-element advantages should sum to ~0"
    )

    print("Exercise 1 PASSED: compute_advantages works correctly")


# =============================================================================
# Exercise 2: Rollout Collection
#
# Implement collect_rollouts: for one scenario, sample N actions from the
# policy and score each one with synthetic_reward.
# Return a list of N Trajectory objects with rewards populated.
# =============================================================================

def collect_rollouts(
    policy: SyntheticPolicy,
    scenario: Scenario,
    n_rollouts: int = 4,
) -> list[Trajectory]:
    """
    Collect N rollouts for a single scenario.

    For each rollout:
    1. Sample an action from the policy
    2. Score it with synthetic_reward
    3. Store as a Trajectory with reward set

    Parameters
    ----------
    policy : SyntheticPolicy
        The current policy to sample from.
    scenario : Scenario
        The scenario to collect rollouts for.
    n_rollouts : int
        Number of rollouts (group size for GRPO).

    Returns
    -------
    list[Trajectory]
        N trajectories with rewards populated.
    """
    # --- Implement this function ---
    trajectories = []
    for _ in range(n_rollouts):
        action = policy.sample_action(scenario.scenario_id)
        reward = synthetic_reward(scenario, action)
        trajectories.append(Trajectory(
            scenario_id=scenario.scenario_id,
            action=action,
            reward=reward,
        ))
    return trajectories


def test_exercise_2():
    """Self-check: rollout collection."""
    random.seed(42)
    scenario = Scenario(scenario_id="test_s1", correct_action=1, difficulty=0.0)
    policy = SyntheticPolicy([scenario], initial_p=0.5)

    trajectories = collect_rollouts(policy, scenario, n_rollouts=4)

    assert len(trajectories) == 4, f"Expected 4 trajectories, got {len(trajectories)}"
    for t in trajectories:
        assert t.scenario_id == "test_s1", "Trajectory scenario_id mismatch"
        assert t.action in (0, 1), f"Invalid action: {t.action}"
        assert t.reward is not None, "Reward must be set"
        assert 0.0 <= t.reward <= 1.0, f"Reward out of range: {t.reward}"

    # With difficulty=0.0 (no noise), action=correct_action always gives reward=1.0
    scenario_easy = Scenario(scenario_id="easy", correct_action=1, difficulty=0.0)
    policy_certain = SyntheticPolicy([scenario_easy], initial_p=1.0)  # Always picks 1
    rollouts_easy = collect_rollouts(policy_certain, scenario_easy, n_rollouts=4)
    for t in rollouts_easy:
        assert t.reward == 1.0, (
            f"With p=1.0 and correct_action=1, reward should be 1.0, got {t.reward}"
        )

    print("Exercise 2 PASSED: collect_rollouts works correctly")


# =============================================================================
# Exercise 3: One Training Step
#
# Implement training_step: given a list of scenarios, collect rollouts,
# compute advantages, and update the policy for each scenario.
# Return the mean reward across all trajectories in this step.
# =============================================================================

def training_step(
    policy: SyntheticPolicy,
    scenarios: list[Scenario],
    n_rollouts: int = 4,
    learning_rate: float = 0.05,
) -> dict:
    """
    Run one training step across a batch of scenarios.

    For each scenario:
    1. Collect n_rollouts trajectories
    2. Compute group advantages
    3. Update the policy for each trajectory using its advantage

    Parameters
    ----------
    policy : SyntheticPolicy
    scenarios : list[Scenario]
        The batch of scenarios for this step.
    n_rollouts : int
        Group size per scenario.
    learning_rate : float
        Passed to policy.update().

    Returns
    -------
    dict with keys:
        "mean_reward" : float — average reward across all trajectories
        "n_trajectories" : int — total trajectories collected
    """
    # --- Implement this function ---
    all_rewards = []

    for scenario in scenarios:
        # Collect rollouts
        trajectories = collect_rollouts(policy, scenario, n_rollouts)

        # Compute advantages for this group
        rewards = [t.reward for t in trajectories]
        advantages = compute_advantages(rewards)
        all_rewards.extend(rewards)

        # Update policy for each trajectory in the group
        for trajectory, advantage in zip(trajectories, advantages):
            policy.update(scenario.scenario_id, trajectory.action, advantage, learning_rate)

    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    return {
        "mean_reward": mean_reward,
        "n_trajectories": len(all_rewards),
    }


def test_exercise_3():
    """Self-check: training step."""
    random.seed(0)
    scenario = Scenario("s1", correct_action=1, difficulty=0.0)
    policy = SyntheticPolicy([scenario], initial_p=0.5)

    result = training_step(policy, [scenario], n_rollouts=4, learning_rate=0.1)

    assert "mean_reward" in result, "Result must contain 'mean_reward'"
    assert "n_trajectories" in result, "Result must contain 'n_trajectories'"
    assert result["n_trajectories"] == 4, (
        f"Expected 4 trajectories, got {result['n_trajectories']}"
    )
    assert 0.0 <= result["mean_reward"] <= 1.0, (
        f"mean_reward out of range: {result['mean_reward']}"
    )

    # After many steps with no noise, policy probability for correct action should increase
    random.seed(1)
    scenario_det = Scenario("det", correct_action=1, difficulty=0.0)
    policy_learn = SyntheticPolicy([scenario_det], initial_p=0.5)
    initial_p = policy_learn.probs["det"]

    for _ in range(20):
        training_step(policy_learn, [scenario_det], n_rollouts=4, learning_rate=0.1)

    final_p = policy_learn.probs["det"]
    assert final_p > initial_p, (
        f"Policy should increase p(action=1) when correct_action=1. "
        f"Initial: {initial_p:.3f}, Final: {final_p:.3f}"
    )

    print("Exercise 3 PASSED: training_step correctly updates the policy")


# =============================================================================
# Exercise 4: Checkpoint Save and Load
#
# Implement save_checkpoint and load_checkpoint functions that persist
# and restore: (a) the policy weights, and (b) the training state
# (step number and reward history).
# =============================================================================

@dataclass
class TrainingState:
    """Serializable training progress."""
    last_completed_step: int = 0
    step_rewards: list[float] = field(default_factory=list)
    best_reward: float = 0.0
    best_step: int = 0


def save_checkpoint(
    policy: SyntheticPolicy,
    state: TrainingState,
    checkpoint_dir: str,
    step: int,
):
    """
    Save policy weights and training state to a step-specific directory.

    Directory structure created:
        {checkpoint_dir}/step_{step:04d}/
            policy_weights.json    — policy.probs dict
            training_state.json    — TrainingState fields

    Also updates {checkpoint_dir}/best/ if this step has the highest reward.

    Parameters
    ----------
    policy : SyntheticPolicy
    state : TrainingState
        Must have step_rewards populated for this step already.
    checkpoint_dir : str
    step : int
    """
    # --- Implement this function ---
    step_dir = Path(checkpoint_dir) / f"step_{step:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    # Save policy weights
    policy.save(str(step_dir))

    # Save training state
    with open(step_dir / "training_state.json", "w") as f:
        json.dump(asdict(state), f, indent=2)

    # Update best/ if this is the best reward so far
    if state.step_rewards and state.step_rewards[-1] >= state.best_reward:
        best_dir = Path(checkpoint_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        # Copy policy weights to best/
        import shutil
        if (best_dir / "policy_weights.json").exists():
            (best_dir / "policy_weights.json").unlink()
        shutil.copy(step_dir / "policy_weights.json", best_dir / "policy_weights.json")
        with open(best_dir / "training_state.json", "w") as f:
            json.dump(asdict(state), f, indent=2)


def load_checkpoint(
    scenarios: list[Scenario],
    checkpoint_dir: str,
    step: Optional[int] = None,
) -> tuple["SyntheticPolicy", TrainingState]:
    """
    Load policy and training state from a checkpoint.

    Parameters
    ----------
    scenarios : list[Scenario]
    checkpoint_dir : str
    step : int or None
        If None, loads from the 'best/' checkpoint.
        If an int, loads from step_{step:04d}/.

    Returns
    -------
    (policy, state) : tuple[SyntheticPolicy, TrainingState]
    """
    # --- Implement this function ---
    if step is None:
        load_dir = Path(checkpoint_dir) / "best"
        label = "best"
    else:
        load_dir = Path(checkpoint_dir) / f"step_{step:04d}"
        label = f"step_{step:04d}"

    if not load_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {load_dir}")

    policy = SyntheticPolicy.load(scenarios, str(load_dir))

    state_path = load_dir / "training_state.json"
    with open(state_path) as f:
        state_data = json.load(f)
    state = TrainingState(**state_data)

    return policy, state


def test_exercise_4():
    """Self-check: checkpoint save and load."""
    random.seed(5)
    scenarios = [
        Scenario("a", correct_action=1, difficulty=0.1),
        Scenario("b", correct_action=0, difficulty=0.2),
    ]
    policy = SyntheticPolicy(scenarios, initial_p=0.7)
    policy.probs["a"] = 0.82   # Simulate post-training probs
    policy.probs["b"] = 0.23

    state = TrainingState(
        last_completed_step=5,
        step_rewards=[0.4, 0.5, 0.55, 0.6, 0.65],
        best_reward=0.65,
        best_step=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_checkpoint(policy, state, tmpdir, step=5)

        # Verify files exist
        step_dir = Path(tmpdir) / "step_0005"
        assert (step_dir / "policy_weights.json").exists(), "policy_weights.json not saved"
        assert (step_dir / "training_state.json").exists(), "training_state.json not saved"

        # Load and verify weights are preserved
        loaded_policy, loaded_state = load_checkpoint(scenarios, tmpdir, step=5)

        assert abs(loaded_policy.probs["a"] - 0.82) < 1e-6, (
            f"Policy prob for 'a' should be 0.82, got {loaded_policy.probs['a']}"
        )
        assert abs(loaded_policy.probs["b"] - 0.23) < 1e-6, (
            f"Policy prob for 'b' should be 0.23, got {loaded_policy.probs['b']}"
        )
        assert loaded_state.last_completed_step == 5, (
            f"Expected step 5, got {loaded_state.last_completed_step}"
        )
        assert loaded_state.best_reward == 0.65, (
            f"Expected best_reward=0.65, got {loaded_state.best_reward}"
        )

        # Load from 'best' checkpoint (should work since step 5 had the best reward)
        best_policy, best_state = load_checkpoint(scenarios, tmpdir, step=None)
        assert abs(best_policy.probs["a"] - 0.82) < 1e-6, (
            "Best checkpoint should have same weights as step 5"
        )

    print("Exercise 4 PASSED: save_checkpoint and load_checkpoint work correctly")


# =============================================================================
# Exercise 5: Full Training Loop with Reward Tracking
#
# Wire up exercises 1–4 into a complete training loop.
# The loop must:
#   - Run for n_steps steps
#   - Record mean_reward at each step
#   - Save a checkpoint every save_every_n_steps steps
#   - Support resuming from a checkpoint
#   - Return the full reward history
# =============================================================================

def run_training_loop(
    scenarios: list[Scenario],
    checkpoint_dir: str,
    n_steps: int = 100,
    n_rollouts: int = 4,
    batch_size: int = 4,
    learning_rate: float = 0.05,
    save_every_n_steps: int = 10,
    resume: bool = False,
) -> list[float]:
    """
    Complete training loop with checkpointing and optional resume.

    Parameters
    ----------
    scenarios : list[Scenario]
    checkpoint_dir : str
        Directory for saving checkpoints.
    n_steps : int
    n_rollouts : int
        Rollouts per scenario per step.
    batch_size : int
        Scenarios sampled per step.
    learning_rate : float
    save_every_n_steps : int
    resume : bool
        If True and checkpoint_dir exists, resume from last checkpoint.

    Returns
    -------
    list[float]
        mean_reward at each step from start_step to n_steps.
    """
    # --- Implement this function ---
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Initialize or resume
    if resume and (Path(checkpoint_dir) / "training_state.json").exists():
        # Load from the most recent step checkpoint
        state_path = Path(checkpoint_dir) / "training_state.json"
        with open(state_path) as f:
            state_data = json.load(f)
        state = TrainingState(**state_data)
        last_step = state.last_completed_step
        last_checkpoint = Path(checkpoint_dir) / f"step_{last_step:04d}"
        if last_checkpoint.exists():
            policy = SyntheticPolicy.load(scenarios, str(last_checkpoint))
            print(f"Resumed from step {last_step} (best reward: {state.best_reward:.4f})")
        else:
            policy = SyntheticPolicy(scenarios)
            state = TrainingState()
            print("Checkpoint not found, starting fresh.")
    else:
        policy = SyntheticPolicy(scenarios)
        state = TrainingState()

    start_step = state.last_completed_step + 1
    step_rewards = list(state.step_rewards)  # History so far

    for step in range(start_step, n_steps + 1):
        # Sample a batch of scenarios
        batch = random.sample(scenarios, k=min(batch_size, len(scenarios)))

        # Run one training step
        result = training_step(policy, batch, n_rollouts, learning_rate)
        mean_reward = result["mean_reward"]
        step_rewards.append(mean_reward)

        # Update training state
        state.last_completed_step = step
        state.step_rewards = step_rewards
        if mean_reward > state.best_reward:
            state.best_reward = mean_reward
            state.best_step = step

        # Save checkpoint
        if step % save_every_n_steps == 0:
            save_checkpoint(policy, state, checkpoint_dir, step)

        # Save top-level state file for resume detection
        with open(Path(checkpoint_dir) / "training_state.json", "w") as f:
            json.dump(asdict(state), f, indent=2)

        if step % 20 == 0 or step == n_steps:
            print(f"Step {step:4d} | mean_reward={mean_reward:.4f} | "
                  f"best={state.best_reward:.4f} @ step {state.best_step}")

    return step_rewards


def test_exercise_5():
    """Self-check: full training loop shows reward improvement."""
    random.seed(99)

    # Create scenarios where action=1 is always correct and difficulty is low
    scenarios = [
        Scenario(f"s{i}", correct_action=1, difficulty=0.05)
        for i in range(10)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        rewards = run_training_loop(
            scenarios=scenarios,
            checkpoint_dir=tmpdir,
            n_steps=80,
            n_rollouts=4,
            batch_size=5,
            learning_rate=0.1,
            save_every_n_steps=20,
        )

        # Reward should improve: compare first 10 steps vs last 10 steps
        early_mean = sum(rewards[:10]) / 10
        late_mean = sum(rewards[-10:]) / 10
        assert late_mean > early_mean, (
            f"Training should improve reward over time. "
            f"Early mean: {early_mean:.4f}, Late mean: {late_mean:.4f}"
        )

        # Final reward should be substantially above 0.5 (random baseline)
        assert late_mean > 0.70, (
            f"After 80 steps on easy scenarios, late mean reward should exceed 0.70. "
            f"Got {late_mean:.4f}. Check policy update logic."
        )

        # Checkpoint files should exist
        checkpoint_dirs = list(Path(tmpdir).glob("step_*"))
        assert len(checkpoint_dirs) >= 4, (
            f"Expected at least 4 checkpoint directories (steps 20, 40, 60, 80), "
            f"found {len(checkpoint_dirs)}"
        )

        # Resume: run 20 more steps from checkpoint
        rewards_resumed = run_training_loop(
            scenarios=scenarios,
            checkpoint_dir=tmpdir,
            n_steps=100,
            n_rollouts=4,
            batch_size=5,
            learning_rate=0.1,
            save_every_n_steps=20,
            resume=True,
        )

        # Resumed run should have more total rewards than the original (extends history)
        assert len(rewards_resumed) > len(rewards), (
            f"Resumed run should have more reward entries than original "
            f"({len(rewards_resumed)} vs {len(rewards)})"
        )

    print("Exercise 5 PASSED: training loop shows reward improvement and resume works")


# =============================================================================
# Bonus: Reward Trend Visualization
#
# Plot the reward history from a completed training run.
# Uses only the standard library (no matplotlib required for the test).
# If matplotlib is available, saves a PNG. Otherwise prints an ASCII chart.
# =============================================================================

def plot_reward_trend(step_rewards: list[float], window: int = 10, save_path: str = None):
    """
    Visualize reward trend from a training run.

    Falls back to ASCII chart if matplotlib is not available.

    Parameters
    ----------
    step_rewards : list[float]
        Mean reward at each step.
    window : int
        Smoothing window for moving average.
    save_path : str or None
        If provided, saves the plot to this path (PNG). Requires matplotlib.
    """
    # Compute moving average
    moving_avg = []
    for i in range(len(step_rewards)):
        start = max(0, i - window + 1)
        avg = sum(step_rewards[start:i+1]) / (i - start + 1)
        moving_avg.append(avg)

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        steps = list(range(1, len(step_rewards) + 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, step_rewards, alpha=0.3, color="#4299e1", label="Step reward")
        ax.plot(steps, moving_avg, color="#2b6cb0", linewidth=2,
                label=f"{window}-step moving average")
        ax.axhline(y=0.5, color="#fc8181", linestyle="--", alpha=0.7,
                   label="Random baseline (0.5)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean RULER reward")
        ax.set_title("Training Progress: Reward Trend")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved reward trend plot to {save_path}")
        else:
            plt.show()

    except ImportError:
        # Fallback: ASCII bar chart
        print("\nReward trend (every 5 steps):")
        print(f"{'Step':>6} {'Reward':>8} {'Moving avg':>12} {'Bar'}")
        print("-" * 50)
        for i in range(0, len(step_rewards), 5):
            r = step_rewards[i]
            avg = moving_avg[i]
            bar_len = int(r * 30)
            bar = "#" * bar_len + "-" * (30 - bar_len)
            print(f"{i+1:>6} {r:>8.4f} {avg:>12.4f} |{bar}|")


# =============================================================================
# Main: run all exercises in sequence
# =============================================================================

def main():
    print("=" * 60)
    print("Module 05 — Training Loop Self-Check Exercises")
    print("=" * 60)

    random.seed(42)

    print("\n--- Exercise 1: Advantage Normalization ---")
    test_exercise_1()

    print("\n--- Exercise 2: Rollout Collection ---")
    test_exercise_2()

    print("\n--- Exercise 3: Training Step ---")
    test_exercise_3()

    print("\n--- Exercise 4: Checkpoint Save and Load ---")
    test_exercise_4()

    print("\n--- Exercise 5: Full Training Loop ---")
    test_exercise_5()

    print("\n--- Bonus: Visualizing a Training Run ---")
    random.seed(7)
    scenarios = [
        Scenario(f"q{i}", correct_action=random.randint(0, 1), difficulty=0.1)
        for i in range(15)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Running 100-step training for visualization...")
        rewards = run_training_loop(
            scenarios=scenarios,
            checkpoint_dir=tmpdir,
            n_steps=100,
            n_rollouts=4,
            batch_size=8,
            learning_rate=0.08,
            save_every_n_steps=10,
        )
        plot_reward_trend(rewards, window=10, save_path="reward_trend.png")

    print("\n" + "=" * 60)
    print("All exercises PASSED.")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Inspect 'reward_trend.png' to see the learning curve")
    print("  - Experiment: change difficulty, learning_rate, n_rollouts")
    print("  - Observe how KL penalty prevents overshooting (try lr=1.0)")
    print("  - Compare: what happens with n_rollouts=1 vs n_rollouts=8?")


if __name__ == "__main__":
    main()

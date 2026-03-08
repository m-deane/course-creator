# Reinforcement Learning Templates

Production-ready Python scaffolds for building, wrapping, and tracking RL experiments.
Copy any file into your project and customise the constants at the top.
All templates are self-contained and runnable with `python <file>.py`.

---

## `rl_agent_template.py`

A complete agent framework built on an abstract `BaseAgent` class that enforces a four-method contract (`act`, `learn`, `save`, `load`).  Two concrete implementations are included: `QLearningAgent` stores a numpy Q-table for fully discrete environments (FrozenLake-v1 out of the box), while `DQNAgent` pairs a two-hidden-layer PyTorch MLP with an experience replay buffer and a hard-update target network for continuous-state environments (CartPole-v1 out of the box).  The `train_agent` function runs the standard episode loop with epsilon decay, periodic checkpointing, and structured logging; `evaluate_agent` switches the agent to deterministic mode and returns mean ± std reward over N evaluation episodes.

```python
import gymnasium as gym
from rl_agent_template import DQNAgent, train_agent, evaluate_agent

env = gym.make("CartPole-v1")
agent = DQNAgent(state_dim=4, action_dim=2, hidden_dim=64, batch_size=64)
history = train_agent(agent, env, episodes=300, checkpoint_dir="./checkpoints")
metrics = evaluate_agent(agent, env, episodes=20)
print(f"Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
```

---

## `environment_wrapper_template.py`

A set of four `gymnasium.Wrapper` subclasses that can be composed freely.
`DiscreteStateWrapper` digitises a continuous `Box` observation into a single integer using equal-width binning and mixed-radix encoding, making any continuous environment compatible with tabular agents.
`RewardShapingWrapper` applies the potential-based shaping formula from Ng et al. (1999) — supply any `phi(obs) -> float` function and the shaped reward provably preserves the original MDP's optimal policy.
`FrameStackWrapper` concatenates the last N observations into one flat vector, giving feedforward agents access to temporal context without recurrence.
`RecordEpisodeStats` accumulates per-episode reward, length, wall-clock time, and user-defined custom metrics, exposing them via `info["episode"]` at each episode boundary.  A `make_wrapped_env` builder composes all four in one call.

```python
from environment_wrapper_template import make_wrapped_env

# Discrete state (for Q-learning) + episode stats recording
env = make_wrapped_env("CartPole-v1", n_bins=8, record_stats=True)

# Frame-stacked continuous state (for DQN)
env = make_wrapped_env("CartPole-v1", n_frames=4, n_bins=0, record_stats=True)
```

---

## `experiment_tracker_template.py`

A self-contained experiment logger that writes all data to a single JSON file — no MLflow, Weights & Biases, or database required.
`ExperimentTracker` exposes `log_hyperparams`, `log_episode`, `save_checkpoint`, `load_checkpoint`, and `load_best_checkpoint`.
`plot_training_curves` produces a multi-panel matplotlib figure with raw rewards, rolling averages, episode lengths, and any custom metrics (e.g. constraint violations).
`ExperimentTracker.compare_runs` overlays rolling-average curves from multiple tracker instances on the same axes for hyperparameter comparison.
The JSON log is written atomically (tmp-file + rename), so a crash mid-run never corrupts previous data.

```python
from experiment_tracker_template import ExperimentTracker

tracker = ExperimentTracker("./runs/dqn_v1", experiment_name="DQN v1")
tracker.log_hyperparams({"lr": 1e-3, "gamma": 0.99, "batch_size": 64})

for episode in range(300):
    # ... your training loop ...
    tracker.log_episode(reward=ep_reward, length=ep_length,
                        info={"constraint_violations": violations})
    if episode % 50 == 0:
        tracker.save_checkpoint(agent, episode)

tracker.plot_training_curves(save_path="./runs/dqn_v1/curves.png",
                             extra_metrics=["constraint_violations"])
print(tracker.summary())
```

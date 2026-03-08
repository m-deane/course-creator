# RL Templates — Quick Reference

| Task | Template | Class / Function | Notes |
|------|----------|-----------------|-------|
| Tabular Q-learning (discrete states) | `rl_agent_template.py` | `QLearningAgent` | Works with FrozenLake-v1; set `state_dim=env.observation_space.n` |
| Deep Q-network (continuous states) | `rl_agent_template.py` | `DQNAgent` | Works with CartPole-v1; requires PyTorch |
| Run a training loop | `rl_agent_template.py` | `train_agent()` | Handles epsilon decay, logging, checkpointing |
| Evaluate a trained agent | `rl_agent_template.py` | `evaluate_agent()` | Returns mean/std/min/max reward; disables exploration |
| Save agent weights to disk | `rl_agent_template.py` | `agent.save(path)` | Q-learning → `.npy`; DQN → `.pt` |
| Load agent weights from disk | `rl_agent_template.py` | `agent.load(path)` | Extension auto-detected |
| Make continuous state compatible with tabular Q | `environment_wrapper_template.py` | `DiscreteStateWrapper` | Set `n_bins` per dimension; observation_space becomes `Discrete(n_bins**obs_dim)` |
| Shape sparse or delayed rewards | `environment_wrapper_template.py` | `RewardShapingWrapper` | Supply `phi(obs) -> float`; set `gamma` to match agent |
| Give MLP agent temporal context | `environment_wrapper_template.py` | `FrameStackWrapper` | Stacks last N frames into one flat vector |
| Log episode rewards and lengths | `environment_wrapper_template.py` | `RecordEpisodeStats` | Stats in `info["episode"]` at episode end |
| Compose all wrappers at once | `environment_wrapper_template.py` | `make_wrapped_env()` | One-liner builder; order is deterministic |
| Log hyperparameters | `experiment_tracker_template.py` | `ExperimentTracker.log_hyperparams()` | Written immediately to JSON |
| Log per-episode statistics | `experiment_tracker_template.py` | `ExperimentTracker.log_episode()` | Accepts optional `info` dict for custom metrics |
| Save agent checkpoint with metadata | `experiment_tracker_template.py` | `ExperimentTracker.save_checkpoint()` | Calls `agent.save()`; records mean reward at save time |
| Restore best checkpoint | `experiment_tracker_template.py` | `ExperimentTracker.load_best_checkpoint()` | Selects by highest rolling-mean reward |
| Plot reward and length curves | `experiment_tracker_template.py` | `ExperimentTracker.plot_training_curves()` | Rolling average overlay; saves PNG/PDF/SVG |
| Compare multiple runs on one plot | `experiment_tracker_template.py` | `ExperimentTracker.compare_runs()` | Pass a list of tracker instances |
| Print experiment summary | `experiment_tracker_template.py` | `ExperimentTracker.summary()` | Prints reward stats, hyperparams, best checkpoint |
| Resume a crashed experiment | `experiment_tracker_template.py` | `ExperimentTracker(existing_dir)` | Auto-loads JSON log on construction |

## Typical Composition Patterns

### Tabular Q-learning pipeline

```python
from environment_wrapper_template import make_wrapped_env
from rl_agent_template import QLearningAgent, train_agent, evaluate_agent
from experiment_tracker_template import ExperimentTracker

env = make_wrapped_env("CartPole-v1", n_bins=8, record_stats=True)
agent = QLearningAgent(state_dim=env.observation_space.n, action_dim=env.action_space.n)

tracker = ExperimentTracker("./runs/qlearning", "Q-Learning CartPole")
tracker.log_hyperparams({"n_bins": 8, "lr": 0.1, "gamma": 0.99})

history = train_agent(agent, env, episodes=1000)
for ep_reward, ep_length in zip(history["episode_rewards"], history["episode_lengths"]):
    tracker.log_episode(reward=ep_reward, length=ep_length)

tracker.plot_training_curves()
print(tracker.summary())
```

### DQN pipeline

```python
from environment_wrapper_template import make_wrapped_env
from rl_agent_template import DQNAgent, train_agent, evaluate_agent
from experiment_tracker_template import ExperimentTracker

env = make_wrapped_env("CartPole-v1", n_frames=4, n_bins=0, record_stats=True)
obs_dim = env.observation_space.shape[0]   # 4 * 4 = 16 with frame stacking
agent = DQNAgent(state_dim=obs_dim, action_dim=env.action_space.n, hidden_dim=128)

tracker = ExperimentTracker("./runs/dqn", "DQN CartPole")
tracker.log_hyperparams({"lr": 1e-3, "gamma": 0.99, "batch_size": 64, "n_frames": 4})

history = train_agent(agent, env, episodes=300, checkpoint_dir="./runs/dqn/checkpoints",
                      checkpoint_interval=50)
for ep_reward, ep_length in zip(history["episode_rewards"], history["episode_lengths"]):
    tracker.log_episode(reward=ep_reward, length=ep_length)

metrics = evaluate_agent(agent, env, episodes=20)
print(f"Eval: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
tracker.plot_training_curves(save_path="./runs/dqn/curves.png")
```

### Hyperparameter comparison

```python
from experiment_tracker_template import ExperimentTracker

t1 = ExperimentTracker("./runs/lr_1e3", "lr=1e-3")
t2 = ExperimentTracker("./runs/lr_5e4", "lr=5e-4")
# ... fill each tracker with training data ...
ExperimentTracker.compare_runs([t1, t2], metric="reward", save_path="./comparison.png")
```

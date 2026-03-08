# Reinforcement Learning Recipes

Copy-paste code patterns for common RL tasks. Each recipe is a standalone,
working function you can drop directly into your own project.

## Files

### `common_patterns.py`
Core algorithmic building blocks used across RL methods:

| Pattern | Description |
|---|---|
| `epsilon_greedy_action` | Action selection with optional exponential decay |
| `ExperienceReplayBuffer` | Fixed-capacity circular buffer with uniform sampling |
| `soft_update_target_network` | Polyak averaging for stable target networks |
| `hard_update_target_network` | Periodic full weight copy for target networks |
| `compute_discounted_returns` | Monte Carlo return calculation from a reward sequence |
| `compute_gae` | Generalized Advantage Estimation (Schulman et al., 2015) |
| `linear_lr_schedule` | Linear learning rate decay over training steps |
| `cosine_lr_schedule` | Cosine annealing learning rate schedule |
| `RunningNormalizer` | Online reward normalization via Welford's algorithm |

### `evaluation_recipes.py`
Evaluation and analysis utilities for trained policies:

| Function | Description |
|---|---|
| `evaluate_policy` | Run N episodes and return mean/std/min/max reward |
| `compare_agents` | Side-by-side performance table for multiple agents |
| `plot_learning_curves` | Smoothed multi-agent reward plot with confidence bands |
| `compute_sample_efficiency` | Episodes required to first exceed a reward threshold |
| `statistical_significance_test` | Welch's t-test on two sets of episode rewards |
| `render_policy_table` | ASCII visualization of a greedy policy over a gridworld |

## Usage

Each function works independently. Copy only what you need:

```python
# Example: grab just the replay buffer
from recipes.common_patterns import ExperienceReplayBuffer

buffer = ExperienceReplayBuffer(capacity=10_000, obs_dim=4, action_dim=1)
buffer.push(obs, action, reward, next_obs, done)
batch = buffer.sample(batch_size=64)
```

## Dependencies

- `common_patterns.py` — `numpy` only
- `evaluation_recipes.py` — `numpy`, `matplotlib`, `scipy` (stats test only)

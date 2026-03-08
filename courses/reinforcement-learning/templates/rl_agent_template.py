"""
RL Agent Template — Base framework for tabular and deep reinforcement learning agents.
Works with: Any Gymnasium-compatible environment
Time to working: 10 minutes

Included agents:
- QLearningAgent: tabular Q-learning for discrete state/action spaces
- DQNAgent: deep Q-network with experience replay and target network (PyTorch)

Example use cases:
- CartPole-v1 balancing (continuous state, discrete actions)
- FrozenLake-v1 navigation (discrete state and actions)
- Custom trading or scheduling environments
"""

from __future__ import annotations

import abc
import collections
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# ============================================================================
# CONSTANTS — customize these for your experiment
# ============================================================================

# Q-Learning defaults
Q_LEARNING_RATE: float = 0.1
Q_DISCOUNT_FACTOR: float = 0.99
Q_EPSILON_START: float = 1.0
Q_EPSILON_END: float = 0.01
Q_EPSILON_DECAY: float = 0.995

# DQN defaults
DQN_LEARNING_RATE: float = 1e-3
DQN_DISCOUNT_FACTOR: float = 0.99
DQN_EPSILON_START: float = 1.0
DQN_EPSILON_END: float = 0.01
DQN_EPSILON_DECAY: float = 0.995
DQN_BATCH_SIZE: int = 64
DQN_REPLAY_BUFFER_SIZE: int = 10_000
DQN_TARGET_UPDATE_FREQ: int = 100  # steps between target network syncs
DQN_HIDDEN_DIM: int = 128

# Training defaults
DEFAULT_EPISODES: int = 500
LOG_INTERVAL: int = 50  # print progress every N episodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIENCE TUPLE
# ============================================================================


@dataclass
class Experience:
    """A single transition stored in the replay buffer.

    Attributes:
        state: Observation from the environment before the action.
        action: Index of the action taken.
        reward: Scalar reward received after the action.
        next_state: Observation received after the action.
        done: Whether the episode terminated or was truncated.
    """

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ============================================================================
# ABSTRACT BASE AGENT
# ============================================================================


class BaseAgent(abc.ABC):
    """Abstract base class for all RL agents.

    Subclasses must implement:
        act(state)      — select an action given the current observation
        learn(exp)      — update internal parameters from one experience
        save(path)      — persist the agent to disk
        load(path)      — restore the agent from disk
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        """Initialise the agent.

        Args:
            state_dim: Dimensionality of the (flat) observation vector.
            action_dim: Number of discrete actions available.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_steps: int = 0
        self.episodes_done: int = 0

    @abc.abstractmethod
    def act(self, state: np.ndarray) -> int:
        """Select an action for the given state (may be stochastic).

        Args:
            state: Current environment observation as a 1-D numpy array.

        Returns:
            Integer action index in [0, action_dim).
        """

    @abc.abstractmethod
    def learn(self, experience: Experience) -> Optional[float]:
        """Update the agent's parameters from a single transition.

        Args:
            experience: The (s, a, r, s', done) transition to learn from.

        Returns:
            Loss value (float) if the agent performed an update, else None.
        """

    @abc.abstractmethod
    def save(self, path: str | os.PathLike) -> None:
        """Persist the agent's learned parameters to disk.

        Args:
            path: File path (without extension) where data will be written.
        """

    @abc.abstractmethod
    def load(self, path: str | os.PathLike) -> None:
        """Restore the agent's learned parameters from disk.

        Args:
            path: File path (without extension) to load from.
        """

    def set_eval_mode(self) -> None:
        """Switch the agent to deterministic (greedy) evaluation mode.

        Override in subclasses that maintain an exploration parameter.
        """

    def set_train_mode(self) -> None:
        """Switch the agent back to stochastic training mode.

        Override in subclasses that maintain an exploration parameter.
        """


# ============================================================================
# TABULAR Q-LEARNING AGENT
# ============================================================================


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent for discrete state and action spaces.

    Implements the classic one-step Q-learning update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    Epsilon is decayed multiplicatively after every episode via ``decay_epsilon()``.

    Args:
        state_dim:      Number of discrete states (flattened if needed).
        action_dim:     Number of discrete actions.
        learning_rate:  Step size alpha for Q-updates.
        discount:       Discount factor gamma in [0, 1].
        epsilon_start:  Initial exploration probability.
        epsilon_end:    Minimum exploration probability.
        epsilon_decay:  Multiplicative decay applied each episode.

    Example::

        env = gym.make("FrozenLake-v1")
        agent = QLearningAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
        )
        train_agent(agent, env, episodes=2000)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = Q_LEARNING_RATE,
        discount: float = Q_DISCOUNT_FACTOR,
        epsilon_start: float = Q_EPSILON_START,
        epsilon_end: float = Q_EPSILON_END,
        epsilon_decay: float = Q_EPSILON_DECAY,
    ) -> None:
        super().__init__(state_dim, action_dim)
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self._eval_mode = False

        # Q-table: rows = states, cols = actions, initialised to zero
        self.q_table: np.ndarray = np.zeros((state_dim, action_dim), dtype=np.float64)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        """Choose an action via epsilon-greedy policy.

        Args:
            state: Current state index (scalar or 1-element array).

        Returns:
            Action index.
        """
        s = int(state)
        if not self._eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        return int(np.argmax(self.q_table[s]))

    def learn(self, experience: Experience) -> Optional[float]:
        """Apply one Q-learning update step.

        Args:
            experience: Transition (s, a, r, s', done).

        Returns:
            Squared TD error (float) for logging purposes.
        """
        s = int(experience.state)
        a = experience.action
        r = experience.reward
        s_next = int(experience.next_state)

        best_next = 0.0 if experience.done else float(np.max(self.q_table[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.lr * td_error
        self.total_steps += 1
        return float(td_error ** 2)

    def save(self, path: str | os.PathLike) -> None:
        """Save Q-table to a .npy file.

        Args:
            path: Destination path (the ``.npy`` extension is appended automatically).
        """
        dest = Path(path).with_suffix(".npy")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, self.q_table)
        logger.info("Q-table saved to %s", dest)

    def load(self, path: str | os.PathLike) -> None:
        """Load Q-table from a .npy file.

        Args:
            path: Source path (with or without ``.npy`` extension).
        """
        src = Path(path).with_suffix(".npy")
        self.q_table = np.load(src)
        logger.info("Q-table loaded from %s", src)

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Decay epsilon by the configured multiplicative factor (call once per episode)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        """Freeze exploration (epsilon = 0) for evaluation."""
        self._eval_mode = True

    def set_train_mode(self) -> None:
        """Re-enable epsilon-greedy exploration for training."""
        self._eval_mode = False


# ============================================================================
# REPLAY BUFFER
# ============================================================================


class ReplayBuffer:
    """Fixed-size circular experience replay buffer.

    Args:
        capacity: Maximum number of transitions to store.

    Example::

        buf = ReplayBuffer(capacity=10_000)
        buf.push(experience)
        batch = buf.sample(64)
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Experience] = collections.deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        """Add a transition to the buffer.

        Args:
            experience: The (s, a, r, s', done) transition.
        """
        self._buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Experience objects.

        Raises:
            ValueError: If the buffer contains fewer transitions than ``batch_size``.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} transitions; need at least {batch_size}."
            )
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)


# ============================================================================
# DQN AGENT (PyTorch)
# ============================================================================


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and a target network.

    Architecture: MLP with two hidden layers of size ``hidden_dim``.
    The online network is trained every step after the buffer warms up.
    The target network is hard-copied from the online network every
    ``target_update_freq`` steps.

    Args:
        state_dim:          Dimensionality of the (flat) continuous observation.
        action_dim:         Number of discrete actions.
        learning_rate:      Adam optimiser learning rate.
        discount:           Discount factor gamma.
        epsilon_start:      Initial epsilon for exploration.
        epsilon_end:        Minimum epsilon.
        epsilon_decay:      Multiplicative decay applied every step.
        batch_size:         Mini-batch size for gradient updates.
        buffer_size:        Replay buffer capacity.
        target_update_freq: Steps between target network hard updates.
        hidden_dim:         Units in each hidden layer.

    Example::

        env = gym.make("CartPole-v1")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        agent = DQNAgent(state_dim=obs_dim, action_dim=act_dim)
        train_agent(agent, env, episodes=300)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = DQN_LEARNING_RATE,
        discount: float = DQN_DISCOUNT_FACTOR,
        epsilon_start: float = DQN_EPSILON_START,
        epsilon_end: float = DQN_EPSILON_END,
        epsilon_decay: float = DQN_EPSILON_DECAY,
        batch_size: int = DQN_BATCH_SIZE,
        buffer_size: int = DQN_REPLAY_BUFFER_SIZE,
        target_update_freq: int = DQN_TARGET_UPDATE_FREQ,
        hidden_dim: int = DQN_HIDDEN_DIM,
    ) -> None:
        super().__init__(state_dim, action_dim)

        # Lazy import so the file is importable without torch when using Q-learning only
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for DQNAgent. Install it with: pip install torch"
            ) from exc

        self._torch = torch
        self._nn = nn

        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._eval_mode = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DQNAgent using device: %s", self.device)

        # Build online and target networks
        self.online_net = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = self._build_network(state_dim, action_dim, hidden_dim).to(self.device)
        self._sync_target()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — robust to outliers

        self.buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Network factory
    # ------------------------------------------------------------------

    def _build_network(self, in_dim: int, out_dim: int, hidden: int) -> "torch.nn.Module":
        """Construct a two-hidden-layer MLP.

        Args:
            in_dim:  Input dimensionality.
            out_dim: Output dimensionality (number of Q-values).
            hidden:  Width of each hidden layer.

        Returns:
            ``torch.nn.Sequential`` model.
        """
        nn = self._nn
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def _sync_target(self) -> None:
        """Hard-copy online network weights into the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> int:
        """Select an action via epsilon-greedy policy.

        Args:
            state: Continuous observation vector (1-D numpy array).

        Returns:
            Action index.
        """
        torch = self._torch
        if not self._eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(s)
            return int(q_values.argmax(dim=1).item())

    def learn(self, experience: Experience) -> Optional[float]:
        """Push transition to buffer; run one gradient update if buffer is ready.

        Args:
            experience: The latest (s, a, r, s', done) transition.

        Returns:
            Training loss (float) if an update was performed, else None.
        """
        torch = self._torch
        self.buffer.push(experience)
        self.total_steps += 1

        # Decay epsilon every step
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Sync target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self._sync_target()

        # Wait until buffer has enough samples
        if len(self.buffer) < self.batch_size:
            return None

        # Sample mini-batch
        batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([float(e.done) for e in batch]).unsqueeze(1).to(self.device)

        # Current Q-values for chosen actions
        current_q = self.online_net(states).gather(1, actions)

        # Target Q-values (no gradient through target net)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        return float(loss.item())

    def save(self, path: str | os.PathLike) -> None:
        """Save the online network's state dict to disk.

        Args:
            path: Destination path (the ``.pt`` extension is appended automatically).
        """
        torch = self._torch
        dest = Path(path).with_suffix(".pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            dest,
        )
        logger.info("DQN checkpoint saved to %s", dest)

    def load(self, path: str | os.PathLike) -> None:
        """Load a previously saved checkpoint.

        Args:
            path: Source path (with or without ``.pt`` extension).
        """
        torch = self._torch
        src = Path(path).with_suffix(".pt")
        checkpoint = torch.load(src, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.total_steps = checkpoint.get("total_steps", 0)
        logger.info("DQN checkpoint loaded from %s", src)

    def set_eval_mode(self) -> None:
        """Freeze epsilon and switch networks to eval mode."""
        self._eval_mode = True
        self.online_net.eval()

    def set_train_mode(self) -> None:
        """Re-enable exploration and switch networks to train mode."""
        self._eval_mode = False
        self.online_net.train()


# ============================================================================
# TRAINING LOOP
# ============================================================================


def train_agent(
    agent: BaseAgent,
    env: gym.Env,
    episodes: int = DEFAULT_EPISODES,
    max_steps_per_episode: int = 1000,
    log_interval: int = LOG_INTERVAL,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 100,
) -> Dict[str, List[float]]:
    """Train an agent in a Gymnasium environment for a fixed number of episodes.

    For ``QLearningAgent`` the epsilon decay is called at the end of each episode.
    For ``DQNAgent`` epsilon is decayed inside ``learn()`` at every step.

    Args:
        agent:                  The agent to train (QLearningAgent or DQNAgent).
        env:                    A Gymnasium environment instance.
        episodes:               Total number of episodes to run.
        max_steps_per_episode:  Hard cap on steps per episode to avoid infinite loops.
        log_interval:           Print progress summary every this many episodes.
        checkpoint_dir:         If set, save checkpoints here at ``checkpoint_interval``.
        checkpoint_interval:    Episodes between each checkpoint save.

    Returns:
        Dictionary with keys ``"episode_rewards"`` and ``"episode_lengths"``
        containing per-episode lists.

    Example::

        env = gym.make("CartPole-v1")
        agent = DQNAgent(state_dim=4, action_dim=2)
        history = train_agent(agent, env, episodes=300)
    """
    agent.set_train_mode()
    history: Dict[str, List[float]] = {"episode_rewards": [], "episode_lengths": []}
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss_sum = 0.0
        update_count = 0

        for step in range(max_steps_per_episode):
            action = agent.act(np.array(state, dtype=np.float32))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            exp = Experience(
                state=np.array(state, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_state=np.array(next_state, dtype=np.float32),
                done=done,
            )
            loss = agent.learn(exp)
            if loss is not None:
                episode_loss_sum += loss
                update_count += 1

            episode_reward += float(reward)
            state = next_state

            if done:
                break

        # Decay epsilon for tabular agents (DQN decays per-step in learn())
        if isinstance(agent, QLearningAgent):
            agent.decay_epsilon()

        agent.episodes_done += 1
        history["episode_rewards"].append(episode_reward)
        history["episode_lengths"].append(step + 1)

        if episode % log_interval == 0:
            recent = history["episode_rewards"][-log_interval:]
            avg_reward = float(np.mean(recent))
            avg_loss = episode_loss_sum / update_count if update_count else float("nan")
            epsilon = getattr(agent, "epsilon", float("nan"))
            elapsed = time.time() - start_time
            logger.info(
                "Episode %4d/%d | avg_reward=%6.2f | loss=%.4f | eps=%.3f | elapsed=%.1fs",
                episode,
                episodes,
                avg_reward,
                avg_loss,
                epsilon,
                elapsed,
            )

        if checkpoint_dir and episode % checkpoint_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_ep{episode}"
            agent.save(ckpt_path)

    logger.info(
        "Training complete. Final 50-episode avg reward: %.2f",
        float(np.mean(history["episode_rewards"][-50:])),
    )
    return history


# ============================================================================
# EVALUATION LOOP
# ============================================================================


def evaluate_agent(
    agent: BaseAgent,
    env: gym.Env,
    episodes: int = 20,
    max_steps_per_episode: int = 1000,
    render: bool = False,
) -> Dict[str, float]:
    """Evaluate a trained agent deterministically (no exploration).

    The agent is switched to eval mode before evaluation and restored
    to train mode afterwards.

    Args:
        agent:                  A trained agent.
        env:                    A Gymnasium environment (can be a different instance
                                from the one used during training, e.g. with render_mode).
        episodes:               Number of evaluation episodes.
        max_steps_per_episode:  Hard cap per episode.
        render:                 If True, call ``env.render()`` each step.

    Returns:
        Dictionary with ``"mean_reward"``, ``"std_reward"``, ``"min_reward"``,
        ``"max_reward"``, and ``"mean_length"``.

    Example::

        eval_env = gym.make("CartPole-v1", render_mode="human")
        metrics = evaluate_agent(agent, eval_env, episodes=10, render=True)
        print(f"Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    """
    agent.set_eval_mode()
    rewards: List[float] = []
    lengths: List[int] = []

    try:
        for _ in range(episodes):
            state, _ = env.reset()
            ep_reward = 0.0

            for step in range(max_steps_per_episode):
                if render:
                    env.render()
                action = agent.act(np.array(state, dtype=np.float32))
                state, reward, terminated, truncated, _ = env.step(action)
                ep_reward += float(reward)
                if terminated or truncated:
                    break

            rewards.append(ep_reward)
            lengths.append(step + 1)
    finally:
        agent.set_train_mode()

    reward_arr = np.array(rewards)
    return {
        "mean_reward": float(np.mean(reward_arr)),
        "std_reward": float(np.std(reward_arr)),
        "min_reward": float(np.min(reward_arr)),
        "max_reward": float(np.max(reward_arr)),
        "mean_length": float(np.mean(lengths)),
    }


# ============================================================================
# RUNNABLE DEMO
# ============================================================================


def _demo_q_learning() -> None:
    """Train a QLearningAgent on FrozenLake-v1 (fully discrete)."""
    print("\n=== Q-Learning on FrozenLake-v1 ===")
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = QLearningAgent(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=0.8,
        discount=0.95,
    )
    train_agent(agent, env, episodes=1000, log_interval=200)
    metrics = evaluate_agent(agent, env, episodes=100)
    print(f"Eval → mean reward: {metrics['mean_reward']:.3f} | std: {metrics['std_reward']:.3f}")
    env.close()


def _demo_dqn() -> None:
    """Train a DQNAgent on CartPole-v1 (continuous state, discrete actions)."""
    print("\n=== DQN on CartPole-v1 ===")
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=1e-3,
        hidden_dim=64,
        batch_size=64,
        buffer_size=5000,
        target_update_freq=50,
    )
    train_agent(agent, env, episodes=300, log_interval=50)
    metrics = evaluate_agent(agent, env, episodes=20)
    print(f"Eval → mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    env.close()


if __name__ == "__main__":
    _demo_q_learning()
    _demo_dqn()

"""
common_patterns.py
==================
Copy-paste RL building blocks. Each function/class is self-contained.
Dependencies: numpy only.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Epsilon-Greedy Action Selection (with exponential decay)
# ---------------------------------------------------------------------------

def epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    """Select an action using epsilon-greedy exploration.

    With probability ``epsilon`` a random action is chosen; otherwise the
    greedy action (argmax of ``q_values``) is returned.

    Args:
        q_values: 1-D array of action values, shape ``(n_actions,)``.
        epsilon: Current exploration probability in ``[0, 1]``.
        rng: Optional NumPy random generator for reproducibility. When
            ``None`` a module-level default generator is used.

    Returns:
        Integer action index.

    Example::

        # Exponential decay loop
        epsilon = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.995

        for step in range(total_steps):
            action = epsilon_greedy_action(q_values, epsilon)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
    """
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class _Transition(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class _Batch(NamedTuple):
    obs: np.ndarray        # (batch, obs_dim)
    action: np.ndarray     # (batch, action_dim)
    reward: np.ndarray     # (batch,)
    next_obs: np.ndarray   # (batch, obs_dim)
    done: np.ndarray       # (batch,)  dtype=bool


class ExperienceReplayBuffer:
    """Fixed-capacity circular replay buffer with uniform random sampling.

    Stores ``(obs, action, reward, next_obs, done)`` transitions and returns
    NumPy-batched samples ready for gradient computation.

    Args:
        capacity: Maximum number of transitions to store. Oldest transitions
            are overwritten once the buffer is full.
        obs_dim: Dimensionality of a single observation vector.
        action_dim: Dimensionality of a single action vector.
        seed: Optional integer seed for reproducible sampling.

    Example::

        buffer = ExperienceReplayBuffer(capacity=100_000, obs_dim=8, action_dim=2)

        # Collect transitions
        buffer.push(obs, action, reward, next_obs, done)

        # Sample a mini-batch once the buffer has enough data
        if len(buffer) >= 64:
            batch = buffer.sample(64)
            # batch.obs.shape == (64, 8)
            # batch.action.shape == (64, 2)
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        seed: int | None = None,
    ) -> None:
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._buffer: deque[_Transition] = deque(maxlen=capacity)

    # ------------------------------------------------------------------
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self._buffer.append(
            _Transition(
                obs=np.asarray(obs, dtype=np.float32),
                action=np.asarray(action, dtype=np.float32),
                reward=float(reward),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
            )
        )

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> _Batch:
        """Sample a uniformly random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Named tuple with stacked NumPy arrays for each field.

        Raises:
            ValueError: If the buffer contains fewer transitions than
                ``batch_size``.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer has {len(self._buffer)} transitions but "
                f"batch_size={batch_size} was requested."
            )
        indices = self._rng.integers(len(self._buffer), size=batch_size)
        transitions = [self._buffer[i] for i in indices]

        return _Batch(
            obs=np.stack([t.obs for t in transitions]),
            action=np.stack([t.action for t in transitions]),
            reward=np.array([t.reward for t in transitions], dtype=np.float32),
            next_obs=np.stack([t.next_obs for t in transitions]),
            done=np.array([t.done for t in transitions], dtype=bool),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True when the buffer has reached its maximum capacity."""
        return len(self._buffer) == self._capacity


# ---------------------------------------------------------------------------
# Soft Target Network Update (Polyak Averaging)
# ---------------------------------------------------------------------------

def soft_update_target_network(
    online_weights: list[np.ndarray],
    target_weights: list[np.ndarray],
    tau: float,
) -> list[np.ndarray]:
    """Blend online weights into target weights via Polyak averaging.

    Computes ``target = tau * online + (1 - tau) * target`` element-wise.
    Returns updated target weights; modifies ``target_weights`` in place.

    Args:
        online_weights: List of weight arrays from the online (learning) network.
        target_weights: List of weight arrays from the target network.
            Must have the same structure as ``online_weights``.
        tau: Interpolation factor in ``(0, 1]``. Small values (e.g. ``0.005``)
            make the target network track the online network slowly, which
            stabilises training. ``tau=1.0`` is equivalent to a hard update.

    Returns:
        The updated ``target_weights`` list (same objects, modified in place).

    Example::

        # Typical usage inside a training loop (pure-NumPy pseudocode)
        tau = 0.005
        online_params = [layer.weight for layer in online_net.layers]
        target_params = [layer.weight for layer in target_net.layers]

        soft_update_target_network(online_params, target_params, tau)
    """
    if len(online_weights) != len(target_weights):
        raise ValueError(
            f"online_weights has {len(online_weights)} arrays but "
            f"target_weights has {len(target_weights)}."
        )
    for online, target in zip(online_weights, target_weights):
        target[:] = tau * online + (1.0 - tau) * target
    return target_weights


# ---------------------------------------------------------------------------
# Hard Target Network Update (Periodic Copy)
# ---------------------------------------------------------------------------

def hard_update_target_network(
    online_weights: list[np.ndarray],
    target_weights: list[np.ndarray],
) -> list[np.ndarray]:
    """Copy online weights directly into the target network.

    Modifies ``target_weights`` in place. Call every ``update_freq`` steps
    rather than every step.

    Args:
        online_weights: List of weight arrays from the online network.
        target_weights: List of weight arrays from the target network.

    Returns:
        The updated ``target_weights`` list.

    Example::

        TARGET_UPDATE_FREQ = 1000

        for step in range(total_steps):
            # ... training logic ...
            if step % TARGET_UPDATE_FREQ == 0:
                hard_update_target_network(online_params, target_params)
    """
    if len(online_weights) != len(target_weights):
        raise ValueError(
            f"online_weights has {len(online_weights)} arrays but "
            f"target_weights has {len(target_weights)}."
        )
    for online, target in zip(online_weights, target_weights):
        target[:] = online
    return target_weights


# ---------------------------------------------------------------------------
# Discounted Return Computation
# ---------------------------------------------------------------------------

def compute_discounted_returns(
    rewards: list[float] | np.ndarray,
    gamma: float,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Compute discounted cumulative returns for a sequence of rewards.

    For a trajectory of length ``T``, the return at timestep ``t`` is::

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t-1} * r_{T-1}

    Computed in a single backward pass (O(T) time and space).

    Args:
        rewards: Sequence of scalar rewards from a single episode.
        gamma: Discount factor in ``[0, 1]``. Typical values: ``0.99``.
        normalize: When ``True``, standardise the returns to zero mean and
            unit variance (can stabilise policy gradient training).

    Returns:
        1-D float32 array of discounted returns, same length as ``rewards``.

    Example::

        rewards = [1.0, 0.0, 0.0, 1.0, 0.0]
        returns = compute_discounted_returns(rewards, gamma=0.99)
        # returns[0] ≈ 1 + 0 + 0 + 0.99^3 + 0 = 1.970299
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    returns = np.empty(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    if normalize:
        std = returns.std()
        if std > 1e-8:
            returns = (returns - returns.mean()) / std
    return returns.astype(np.float32)


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation (GAE)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    dones: list[bool] | np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute advantages and value targets using Generalized Advantage Estimation.

    Implements the estimator from Schulman et al. (2015),
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation".

    The TD(lambda) advantage at step ``t`` is::

        delta_t  = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t      = delta_t + gamma * lam * (1 - done_t) * A_{t+1}

    Args:
        rewards: Episode rewards, shape ``(T,)``.
        values: Critic value estimates for each state, shape ``(T,)``.
            Must include **one extra bootstrap value** at position ``T``
            (i.e. ``V(s_T)``), so pass an array of length ``T + 1``.
        dones: Boolean done flags per timestep, shape ``(T,)``.
        gamma: Discount factor (e.g. ``0.99``).
        lam: GAE smoothing parameter lambda in ``[0, 1]``.
            ``lam=0`` is pure TD(0); ``lam=1`` is Monte Carlo returns.

    Returns:
        Tuple of ``(advantages, returns)`` both shape ``(T,)``, float32.
        ``returns = advantages + values[:-1]`` and can be used as value targets.

    Example::

        # values has T+1 elements: states s_0..s_{T-1} + bootstrap V(s_T)
        advantages, returns = compute_gae(
            rewards=ep_rewards,
            values=ep_values,   # length T+1
            dones=ep_dones,
            gamma=0.99,
            lam=0.95,
        )
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)

    T = len(rewards)
    if len(values) != T + 1:
        raise ValueError(
            f"values must have length T+1={T+1} (including bootstrap), "
            f"got {len(values)}."
        )

    advantages = np.empty(T, dtype=np.float64)
    last_gae = 0.0

    for t in range(T - 1, -1, -1):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:T]
    return advantages.astype(np.float32), returns.astype(np.float32)


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

def linear_lr_schedule(
    initial_lr: float,
    final_lr: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Linearly interpolate the learning rate between two values.

    Returns ``initial_lr`` at step 0 and ``final_lr`` at step ``total_steps``.
    Values are clamped so the result never exceeds ``[final_lr, initial_lr]``.

    Args:
        initial_lr: Starting learning rate (e.g. ``3e-4``).
        final_lr: Ending learning rate (e.g. ``0.0``).
        current_step: Current training iteration (0-indexed).
        total_steps: Total number of training iterations.

    Returns:
        Scalar learning rate for the current step.

    Example::

        optimizer_lr = linear_lr_schedule(3e-4, 0.0, step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = optimizer_lr
    """
    fraction = min(current_step / max(total_steps, 1), 1.0)
    return initial_lr + fraction * (final_lr - initial_lr)


def cosine_lr_schedule(
    initial_lr: float,
    final_lr: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Cosine annealing learning rate schedule.

    Smoothly decays from ``initial_lr`` to ``final_lr`` following a half-cosine
    curve. Stays at ``final_lr`` once ``current_step >= total_steps``.

    Args:
        initial_lr: Starting learning rate.
        final_lr: Minimum learning rate at the end of the schedule.
        current_step: Current training iteration (0-indexed).
        total_steps: Total number of iterations over which to decay.

    Returns:
        Scalar learning rate for the current step.

    Example::

        lr = cosine_lr_schedule(3e-4, 1e-6, step, 500_000)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    """
    fraction = min(current_step / max(total_steps, 1), 1.0)
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * fraction))
    return final_lr + cosine_decay * (initial_lr - final_lr)


# ---------------------------------------------------------------------------
# Reward Normalization (Running Mean / Std via Welford's Algorithm)
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Online reward normalization using Welford's one-pass algorithm.

    Maintains a running estimate of the mean and variance of a scalar signal
    and normalizes new values to approximately zero mean / unit variance.
    Safe against divide-by-zero when variance is near zero.

    Args:
        clip: If positive, normalized values are clipped to ``[-clip, clip]``.
            Setting ``clip=5.0`` is a common choice to prevent large outliers.
        epsilon: Small constant added to the standard deviation for numerical
            stability (default ``1e-8``).

    Example::

        normalizer = RunningNormalizer(clip=5.0)

        for step in range(total_steps):
            raw_reward = env.step(action)[1]
            normalizer.update(raw_reward)
            norm_reward = normalizer.normalize(raw_reward)
            # Use norm_reward for training; raw_reward for logging
    """

    def __init__(self, clip: float = 0.0, epsilon: float = 1e-8) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0   # sum of squared deviations (Welford)
        self.clip = clip
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    def update(self, value: float) -> None:
        """Incorporate a new scalar observation into the running statistics."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    # ------------------------------------------------------------------
    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def std(self) -> float:
        """Current running standard deviation (population estimate)."""
        if self._count < 2:
            return 1.0
        return float(np.sqrt(self._m2 / self._count))

    # ------------------------------------------------------------------
    def normalize(self, value: float) -> float:
        """Normalize a scalar value using the current running statistics.

        Args:
            value: Raw scalar (e.g. a reward).

        Returns:
            Normalized value; optionally clipped.
        """
        normed = (value - self._mean) / (self.std + self.epsilon)
        if self.clip > 0.0:
            normed = float(np.clip(normed, -self.clip, self.clip))
        return normed

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"RunningNormalizer(n={self._count}, mean={self.mean:.4f}, "
            f"std={self.std:.4f}, clip={self.clip})"
        )

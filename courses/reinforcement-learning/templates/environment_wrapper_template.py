"""
Environment Wrapper Template — Gymnasium wrapper toolkit for RL experiments.
Works with: Any Gymnasium-compatible environment
Time to working: 5 minutes

Included wrappers:
- DiscreteStateWrapper:  Digitises continuous observations into a flat integer index
- RewardShapingWrapper:  Adds potential-based reward shaping with a user-supplied function
- FrameStackWrapper:     Stacks the last N observations into a single observation vector
- RecordEpisodeStats:    Accumulates per-episode reward, length, and custom metrics

Example use cases:
- Preparing a continuous-state env for a tabular Q-learning agent
- Shaping sparse rewards to accelerate learning
- Providing temporal context to an MLP agent (frame stacking)
- Logging detailed training statistics without modifying the agent
"""

from __future__ import annotations

import collections
import time
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_N_BINS: int = 10          # Bins per continuous dimension
DEFAULT_STACK_SIZE: int = 4       # Number of frames to stack
POTENTIAL_SHAPING_GAMMA: float = 0.99  # Must match the agent's discount factor


# ============================================================================
# DISCRETE STATE WRAPPER
# ============================================================================


class DiscreteStateWrapper(gym.ObservationWrapper):
    """Digitise a continuous ``Box`` observation space into a single integer.

    Each dimension of the observation is independently binned into ``n_bins``
    equal-width intervals (clipped at the low/high bounds).  The resulting
    bin indices are combined into a single integer using a mixed-radix
    encoding, making the output suitable for tabular methods.

    Args:
        env:          A Gymnasium environment with a ``Box`` observation space.
        n_bins:       Number of bins per observation dimension.
        low_override: Optional array overriding the low bound for each
                      dimension (useful when the space reports ``-inf``).
        high_override: Optional array overriding the high bound.

    Raises:
        TypeError: If the environment's observation space is not a ``Box``.

    Example::

        base_env = gym.make("CartPole-v1")
        env = DiscreteStateWrapper(base_env, n_bins=10)
        # env.observation_space is now Discrete(10**4 = 10000)
        state, _ = env.reset()
        print(state)  # integer in [0, 9999]
    """

    def __init__(
        self,
        env: gym.Env,
        n_bins: int = DEFAULT_N_BINS,
        low_override: Optional[np.ndarray] = None,
        high_override: Optional[np.ndarray] = None,
    ) -> None:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"DiscreteStateWrapper requires a Box observation space, "
                f"got {type(env.observation_space).__name__}."
            )
        super().__init__(env)

        self.n_bins = n_bins
        obs_space: gym.spaces.Box = env.observation_space
        self._obs_dim: int = int(np.prod(obs_space.shape))

        # Allow caller to override unbounded dimensions (e.g. CartPole velocity)
        low = low_override if low_override is not None else obs_space.low.flatten()
        high = high_override if high_override is not None else obs_space.high.flatten()

        # Replace infinite bounds with large finite values
        low = np.where(np.isfinite(low), low, -10.0)
        high = np.where(np.isfinite(high), high, 10.0)

        # Build bin edges per dimension
        self._bins: List[np.ndarray] = [
            np.linspace(low[i], high[i], n_bins + 1)[1:-1]  # interior edges only
            for i in range(self._obs_dim)
        ]

        # Mixed-radix multipliers for encoding: state = sum_i(bin_i * mult_i)
        self._multipliers: np.ndarray = np.array(
            [n_bins ** i for i in range(self._obs_dim)], dtype=np.int64
        )

        total_states: int = n_bins ** self._obs_dim
        self.observation_space = gym.spaces.Discrete(total_states)

    def observation(self, obs: np.ndarray) -> int:
        """Convert a continuous observation vector to a discrete integer.

        Args:
            obs: Raw observation from the wrapped environment.

        Returns:
            Integer state index in ``[0, n_bins**obs_dim)``.
        """
        flat = np.asarray(obs, dtype=np.float64).flatten()
        bin_indices = np.array(
            [int(np.digitize(flat[i], self._bins[i])) for i in range(self._obs_dim)],
            dtype=np.int64,
        )
        return int(np.dot(bin_indices, self._multipliers))


# ============================================================================
# REWARD SHAPING WRAPPER
# ============================================================================


class RewardShapingWrapper(gym.RewardWrapper):
    """Add potential-based reward shaping to any Gymnasium environment.

    Implements the theoretical guarantee from Ng et al. (1999): the shaped
    reward ``r' = r + gamma * phi(s') - phi(s)`` preserves the optimal policy
    of the original MDP when ``phi`` is an arbitrary potential function.

    Args:
        env:               The environment to wrap.
        potential_fn:      A callable ``phi(obs) -> float`` mapping an
                           observation to a scalar potential.
        gamma:             Discount factor used in the shaping formula.
                           Must match the agent's discount factor.
        shaping_weight:    Scale factor applied to the shaping bonus.

    Example::

        def upright_potential(obs):
            # Reward for keeping the pole upright (CartPole)
            pole_angle = obs[2]  # radians from vertical
            return 1.0 - abs(pole_angle) / 0.2095  # 1 at upright, 0 at threshold

        base_env = gym.make("CartPole-v1")
        env = RewardShapingWrapper(base_env, potential_fn=upright_potential, gamma=0.99)
    """

    def __init__(
        self,
        env: gym.Env,
        potential_fn: Callable[[np.ndarray], float],
        gamma: float = POTENTIAL_SHAPING_GAMMA,
        shaping_weight: float = 1.0,
    ) -> None:
        super().__init__(env)
        self._potential_fn = potential_fn
        self._gamma = gamma
        self._shaping_weight = shaping_weight
        self._previous_potential: Optional[float] = None
        self._current_obs: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict]:
        """Reset the environment and initialise the potential baseline.

        Returns:
            Tuple of (initial observation, info dict).
        """
        obs, info = self.env.reset(**kwargs)
        self._current_obs = np.asarray(obs, dtype=np.float64)
        self._previous_potential = float(self._potential_fn(self._current_obs))
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment and add the shaping bonus.

        Args:
            action: Action to take.

        Returns:
            Standard Gymnasium (obs, reward, terminated, truncated, info) tuple
            with a shaping bonus added to reward.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = np.asarray(obs, dtype=np.float64)
        shaped_reward = self.reward(float(reward))
        return obs, shaped_reward, terminated, truncated, info

    def reward(self, reward: float) -> float:
        """Compute shaped reward using the potential-based formula.

        Args:
            reward: Original environment reward.

        Returns:
            Shaped reward ``r + weight * (gamma * phi(s') - phi(s))``.
        """
        if self._current_obs is None or self._previous_potential is None:
            return reward

        current_potential = float(self._potential_fn(self._current_obs))
        shaping_bonus = self._shaping_weight * (
            self._gamma * current_potential - self._previous_potential
        )
        self._previous_potential = current_potential
        return reward + shaping_bonus


# ============================================================================
# FRAME STACK WRAPPER
# ============================================================================


class FrameStackWrapper(gym.ObservationWrapper):
    """Stack the last ``n_frames`` observations into a single flat vector.

    This gives the agent temporal information without requiring recurrence.
    On reset, the initial frame is replicated ``n_frames`` times to fill
    the stack.  The resulting observation space is a ``Box`` whose shape is
    ``(n_frames * original_obs_dim,)``.

    Args:
        env:      A Gymnasium environment with a flat ``Box`` observation space.
        n_frames: Number of consecutive frames to concatenate.

    Raises:
        TypeError: If the observation space is not a flat ``Box``.

    Example::

        base_env = gym.make("CartPole-v1")
        env = FrameStackWrapper(base_env, n_frames=4)
        # env.observation_space.shape == (16,)
        obs, _ = env.reset()
        print(obs.shape)  # (16,)
    """

    def __init__(self, env: gym.Env, n_frames: int = DEFAULT_STACK_SIZE) -> None:
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                f"FrameStackWrapper requires a Box observation space, "
                f"got {type(env.observation_space).__name__}."
            )
        super().__init__(env)

        self.n_frames = n_frames
        base_space: gym.spaces.Box = env.observation_space
        single_dim: int = int(np.prod(base_space.shape))

        self._frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=n_frames)

        # New observation space: n_frames stacked along the first axis
        stacked_low = np.tile(base_space.low.flatten(), n_frames)
        stacked_high = np.tile(base_space.high.flatten(), n_frames)
        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=base_space.dtype,
        )
        self._single_dim = single_dim

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict]:
        """Reset environment and fill frame buffer with the initial observation.

        Returns:
            Tuple of (stacked observation, info dict).
        """
        obs, info = self.env.reset(**kwargs)
        flat = np.asarray(obs, dtype=np.float32).flatten()
        # Fill all slots with the first frame
        self._frame_buffer.clear()
        for _ in range(self.n_frames):
            self._frame_buffer.append(flat.copy())
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Append new frame and return the stacked observation.

        Args:
            obs: Latest raw observation from the environment.

        Returns:
            Concatenated array of shape ``(n_frames * obs_dim,)``.
        """
        flat = np.asarray(obs, dtype=np.float32).flatten()
        self._frame_buffer.append(flat)
        return np.concatenate(list(self._frame_buffer), axis=0)


# ============================================================================
# EPISODE STATISTICS RECORDER
# ============================================================================


class RecordEpisodeStats(gym.Wrapper):
    """Track episode rewards, lengths, wall-clock time, and custom metrics.

    Statistics are accumulated in ``self.episode_stats`` (a list of dicts)
    and also exposed via ``info["episode"]`` at the end of each episode,
    matching the convention used by Gymnasium's built-in ``RecordEpisodeStatistics``.

    Custom metrics (e.g., ``"constraint_violations"``) can be logged through
    ``env.record_metric(key, value)`` during rollouts.

    Args:
        env:           The environment to wrap.
        buffer_size:   Maximum number of episodes to keep in memory.

    Example::

        base_env = gym.make("CartPole-v1")
        env = RecordEpisodeStats(base_env, buffer_size=500)

        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # After the episode ends, info contains the episode summary:
        print(info.get("episode"))
        # {"reward": 42.0, "length": 42, "time": 1.23, ...}

        # Access all history:
        print(env.episode_stats[-1])
    """

    def __init__(self, env: gym.Env, buffer_size: int = 1000) -> None:
        super().__init__(env)
        self._buffer_size = buffer_size
        self.episode_stats: List[Dict[str, Any]] = []
        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._episode_start_time: float = 0.0
        self._pending_metrics: Dict[str, float] = {}

    def reset(self, **kwargs: Any) -> Tuple[Any, Dict]:
        """Reset the environment and clear per-episode accumulators.

        Returns:
            Standard (obs, info) tuple.
        """
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_start_time = time.perf_counter()
        self._pending_metrics = {}
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Step the environment and accumulate episode statistics.

        Args:
            action: Action to take.

        Returns:
            Standard Gymnasium 5-tuple. When the episode ends, ``info["episode"]``
            contains a dict with ``"reward"``, ``"length"``, ``"time"``, and any
            custom metrics logged via ``record_metric()``.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += float(reward)
        self._episode_length += 1

        if terminated or truncated:
            elapsed = time.perf_counter() - self._episode_start_time
            episode_info: Dict[str, Any] = {
                "reward": self._episode_reward,
                "length": self._episode_length,
                "time": round(elapsed, 4),
                **self._pending_metrics,
            }
            info["episode"] = episode_info

            # Append to history, evicting old entries if needed
            self.episode_stats.append(episode_info)
            if len(self.episode_stats) > self._buffer_size:
                self.episode_stats.pop(0)

        return obs, reward, terminated, truncated, info

    def record_metric(self, key: str, value: float) -> None:
        """Log a custom scalar metric for the current episode.

        Call this at any point during an episode (e.g., after each step).
        Values are accumulated by summation; override ``_aggregate_metric``
        to change the aggregation strategy.

        Args:
            key:   Metric name (e.g. ``"constraint_violations"``).
            value: Scalar value to accumulate.
        """
        self._pending_metrics[key] = self._pending_metrics.get(key, 0.0) + value

    def get_rolling_mean(self, key: str = "reward", window: int = 100) -> Optional[float]:
        """Return the rolling mean of a stat over the last ``window`` episodes.

        Args:
            key:    One of ``"reward"``, ``"length"``, or any custom metric key.
            window: Number of recent episodes to average.

        Returns:
            Rolling mean as a float, or None if no episodes have been recorded.
        """
        if not self.episode_stats:
            return None
        recent = self.episode_stats[-window:]
        values = [ep[key] for ep in recent if key in ep]
        return float(np.mean(values)) if values else None

    def summary(self) -> str:
        """Return a human-readable summary of all recorded episodes.

        Returns:
            Multi-line string with count, mean/std/min/max reward and length.
        """
        if not self.episode_stats:
            return "No episodes recorded."

        rewards = [ep["reward"] for ep in self.episode_stats]
        lengths = [ep["length"] for ep in self.episode_stats]
        lines = [
            "=" * 50,
            f"Episodes recorded : {len(rewards)}",
            f"Reward  — mean: {np.mean(rewards):8.2f}  std: {np.std(rewards):6.2f}"
            f"  min: {np.min(rewards):8.2f}  max: {np.max(rewards):8.2f}",
            f"Length  — mean: {np.mean(lengths):8.2f}  std: {np.std(lengths):6.2f}"
            f"  min: {np.min(lengths):8.2f}  max: {np.max(lengths):8.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE BUILDER
# ============================================================================


def make_wrapped_env(
    env_id: str,
    n_bins: int = DEFAULT_N_BINS,
    n_frames: int = 1,
    potential_fn: Optional[Callable[[np.ndarray], float]] = None,
    gamma: float = POTENTIAL_SHAPING_GAMMA,
    record_stats: bool = True,
    buffer_size: int = 1000,
    **make_kwargs: Any,
) -> gym.Env:
    """Build a wrapped Gymnasium environment in one call.

    Wrappers are applied in the following order (innermost first):
    base -> RecordEpisodeStats -> RewardShapingWrapper -> FrameStackWrapper
    -> DiscreteStateWrapper (if n_bins > 0 and n_frames == 1)

    Args:
        env_id:        Gymnasium environment identifier (e.g. ``"CartPole-v1"``).
        n_bins:        If > 0, wrap with ``DiscreteStateWrapper``.
        n_frames:      If > 1, wrap with ``FrameStackWrapper`` before discretising.
        potential_fn:  If provided, wrap with ``RewardShapingWrapper``.
        gamma:         Discount factor for reward shaping.
        record_stats:  If True, wrap with ``RecordEpisodeStats``.
        buffer_size:   Episode history size for ``RecordEpisodeStats``.
        **make_kwargs: Extra keyword arguments forwarded to ``gym.make``.

    Returns:
        Wrapped Gymnasium environment.

    Example::

        env = make_wrapped_env("CartPole-v1", n_bins=8, record_stats=True)
    """
    env: gym.Env = gym.make(env_id, **make_kwargs)

    if record_stats:
        env = RecordEpisodeStats(env, buffer_size=buffer_size)

    if potential_fn is not None:
        env = RewardShapingWrapper(env, potential_fn=potential_fn, gamma=gamma)

    if n_frames > 1:
        env = FrameStackWrapper(env, n_frames=n_frames)

    if n_bins > 0 and isinstance(env.observation_space, gym.spaces.Box):
        env = DiscreteStateWrapper(env, n_bins=n_bins)

    return env


# ============================================================================
# RUNNABLE DEMO
# ============================================================================


if __name__ == "__main__":
    import textwrap

    # ------------------------------------------------------------------
    # Demo 1: DiscreteStateWrapper
    # ------------------------------------------------------------------
    print("\n--- DiscreteStateWrapper on CartPole-v1 ---")
    base = gym.make("CartPole-v1")
    discrete_env = DiscreteStateWrapper(base, n_bins=6)
    obs, _ = discrete_env.reset()
    print(f"Observation space: {discrete_env.observation_space}")
    print(f"First state (integer): {obs}")
    for _ in range(5):
        obs, r, term, trunc, _ = discrete_env.step(discrete_env.action_space.sample())
        print(f"  state={obs:6d}  reward={r:.1f}  done={term or trunc}")
        if term or trunc:
            break
    discrete_env.close()

    # ------------------------------------------------------------------
    # Demo 2: RewardShapingWrapper
    # ------------------------------------------------------------------
    print("\n--- RewardShapingWrapper on CartPole-v1 ---")

    def pole_angle_potential(obs: np.ndarray) -> float:
        """Potential proportional to how upright the pole is."""
        pole_angle = float(obs[2])
        return max(0.0, 1.0 - abs(pole_angle) / 0.2095)

    base = gym.make("CartPole-v1")
    shaped_env = RewardShapingWrapper(base, potential_fn=pole_angle_potential, gamma=0.99)
    obs, _ = shaped_env.reset()
    total_shaped = 0.0
    for _ in range(20):
        obs, r, term, trunc, _ = shaped_env.step(shaped_env.action_space.sample())
        total_shaped += r
        if term or trunc:
            break
    print(f"Total shaped reward over episode: {total_shaped:.4f}")
    shaped_env.close()

    # ------------------------------------------------------------------
    # Demo 3: FrameStackWrapper
    # ------------------------------------------------------------------
    print("\n--- FrameStackWrapper (4 frames) on CartPole-v1 ---")
    base = gym.make("CartPole-v1")
    stacked_env = FrameStackWrapper(base, n_frames=4)
    obs, _ = stacked_env.reset()
    print(f"Stacked observation shape: {obs.shape}")  # (16,)
    obs, _, _, _, _ = stacked_env.step(0)
    print(f"After one step shape:      {obs.shape}")
    stacked_env.close()

    # ------------------------------------------------------------------
    # Demo 4: RecordEpisodeStats
    # ------------------------------------------------------------------
    print("\n--- RecordEpisodeStats on CartPole-v1 (5 episodes) ---")
    base = gym.make("CartPole-v1")
    stats_env = RecordEpisodeStats(base, buffer_size=100)

    for ep in range(5):
        obs, _ = stats_env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = stats_env.step(
                stats_env.action_space.sample()
            )
            done = terminated or truncated
        print(f"  Episode {ep + 1}: {info.get('episode')}")

    print()
    print(stats_env.summary())
    stats_env.close()

    # ------------------------------------------------------------------
    # Demo 5: make_wrapped_env convenience builder
    # ------------------------------------------------------------------
    print("\n--- make_wrapped_env convenience builder ---")
    env = make_wrapped_env("CartPole-v1", n_bins=8, record_stats=True)
    print(f"Final observation space: {env.observation_space}")
    env.close()
    print("All wrapper demos complete.")

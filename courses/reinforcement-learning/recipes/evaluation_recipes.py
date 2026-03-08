"""
evaluation_recipes.py
=====================
Evaluation and analysis patterns for trained RL policies.
Dependencies: numpy, matplotlib, scipy (stats test only).
"""

from __future__ import annotations

from typing import Protocol, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Protocol: minimal interface expected of agents and environments
# ---------------------------------------------------------------------------

class _Agent(Protocol):
    def act(self, obs: np.ndarray) -> Any: ...


class _Env(Protocol):
    def reset(self) -> np.ndarray: ...
    def step(self, action: Any) -> tuple[np.ndarray, float, bool, dict]: ...


# ---------------------------------------------------------------------------
# Policy Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    agent: _Agent,
    env: _Env,
    n_episodes: int,
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> dict[str, float]:
    """Run the agent for ``n_episodes`` episodes and return summary statistics.

    The agent's ``act`` method is called with the current observation each
    step. The environment must follow the ``gym``-style interface:
    ``reset() -> obs``, ``step(action) -> (obs, reward, done, info)``.

    Args:
        agent: Any object with an ``act(obs) -> action`` method.
        env: Gym-style environment.
        n_episodes: Number of complete episodes to run.
        seed: If provided, ``env.reset(seed=seed + episode_idx)`` is attempted
            for reproducibility (silently ignored if the env does not accept it).
        max_steps: Hard cap on steps per episode to guard against non-terminating
            environments.

    Returns:
        Dictionary with keys:
        ``mean_reward``, ``std_reward``, ``min_reward``, ``max_reward``,
        ``mean_episode_length``, ``n_episodes``.

    Example::

        stats = evaluate_policy(trained_agent, env, n_episodes=100)
        print(f"Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(n_episodes):
        # Attempt seeded reset; fall back to plain reset
        try:
            obs = env.reset(seed=seed + ep if seed is not None else None)
        except TypeError:
            obs = env.reset()

        total_reward = 0.0
        for step in range(max_steps):
            action = agent.act(np.asarray(obs))
            obs, reward, done, *_ = env.step(action)
            total_reward += float(reward)
            if done:
                episode_lengths.append(step + 1)
                break
        else:
            episode_lengths.append(max_steps)

        episode_rewards.append(total_reward)

    rewards_arr = np.array(episode_rewards, dtype=np.float64)
    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "n_episodes": n_episodes,
    }


# ---------------------------------------------------------------------------
# Side-by-Side Agent Comparison
# ---------------------------------------------------------------------------

def compare_agents(
    agents: dict[str, _Agent],
    env: _Env,
    n_episodes: int,
    *,
    seed: int | None = 0,
    max_steps: int = 10_000,
) -> dict[str, dict[str, float]]:
    """Evaluate multiple agents on the same environment and print a comparison table.

    Args:
        agents: Mapping of agent name to agent object. Each agent must
            implement ``act(obs) -> action``.
        env: Shared environment instance. Evaluation is sequential; the
            environment is fully reset between agents.
        n_episodes: Episodes to evaluate per agent.
        seed: Base seed passed to ``evaluate_policy``.
        max_steps: Hard cap on steps per episode.

    Returns:
        Nested dict ``{agent_name: stats_dict}`` where each ``stats_dict``
        matches the output of :func:`evaluate_policy`.

    Example::

        results = compare_agents(
            {"DQN": dqn_agent, "Random": random_agent},
            env,
            n_episodes=50,
        )
        # Also prints a formatted table to stdout.
    """
    results: dict[str, dict[str, float]] = {}
    for name, agent in agents.items():
        results[name] = evaluate_policy(
            agent, env, n_episodes, seed=seed, max_steps=max_steps
        )

    # ---- Print table -------------------------------------------------------
    col_name = max(len(n) for n in results) + 2
    header = (
        f"{'Agent':<{col_name}}"
        f"{'Mean':>10}"
        f"{'Std':>10}"
        f"{'Min':>10}"
        f"{'Max':>10}"
        f"{'Ep Len':>10}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for name, stats in results.items():
        print(
            f"{name:<{col_name}}"
            f"{stats['mean_reward']:>10.2f}"
            f"{stats['std_reward']:>10.2f}"
            f"{stats['min_reward']:>10.2f}"
            f"{stats['max_reward']:>10.2f}"
            f"{stats['mean_episode_length']:>10.1f}"
        )
    print(separator)

    return results


# ---------------------------------------------------------------------------
# Learning Curve Plotting
# ---------------------------------------------------------------------------

def plot_learning_curves(
    rewards_dict: dict[str, list[float] | np.ndarray],
    window: int = 50,
    *,
    ax: plt.Axes | None = None,
    title: str = "Learning Curves",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    alpha_band: float = 0.15,
) -> plt.Axes:
    """Plot smoothed episode rewards for one or more agents.

    Each series is plotted as a rolling-mean line with a +/- 1 rolling-std
    shaded band to show variability.

    Args:
        rewards_dict: Mapping of agent/run name to a 1-D sequence of per-episode
            rewards. Series may have different lengths.
        window: Rolling window size for smoothing (number of episodes).
        ax: Existing Matplotlib ``Axes`` to draw on. A new figure is created
            when ``None``.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        alpha_band: Opacity of the standard-deviation shading band.

    Returns:
        The ``Axes`` object with the completed plot (not yet shown or saved).

    Example::

        ax = plot_learning_curves(
            {"DQN": dqn_rewards, "PPO": ppo_rewards},
            window=100,
        )
        ax.figure.savefig("learning_curves.png", dpi=150)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for label, raw in rewards_dict.items():
        raw_arr = np.asarray(raw, dtype=np.float64)
        T = len(raw_arr)
        if T == 0:
            continue

        # Rolling statistics via cumulative-sum trick (O(T), no padding bias)
        smoothed_mean = np.empty(T)
        smoothed_std = np.empty(T)
        for t in range(T):
            lo = max(0, t - window + 1)
            segment = raw_arr[lo : t + 1]
            smoothed_mean[t] = segment.mean()
            smoothed_std[t] = segment.std() if len(segment) > 1 else 0.0

        episodes = np.arange(T)
        (line,) = ax.plot(episodes, smoothed_mean, linewidth=1.8, label=label)
        ax.fill_between(
            episodes,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=line.get_color(),
            alpha=alpha_band,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# Sample Efficiency
# ---------------------------------------------------------------------------

def compute_sample_efficiency(
    rewards: list[float] | np.ndarray,
    threshold: float,
    *,
    window: int = 1,
) -> int | None:
    """Return the first episode at which the agent surpasses ``threshold``.

    Optionally uses a rolling mean over ``window`` episodes so that isolated
    lucky episodes do not trigger early stopping.

    Args:
        rewards: 1-D sequence of per-episode rewards.
        threshold: Performance threshold to reach.
        window: Number of consecutive episodes whose rolling average must
            exceed ``threshold``. Set to ``1`` for a single-episode check.

    Returns:
        0-indexed episode number (the *last* episode of the qualifying window),
        or ``None`` if the threshold was never reached.

    Example::

        ep = compute_sample_efficiency(rewards, threshold=195.0, window=100)
        if ep is not None:
            print(f"Solved after {ep + 1} episodes")
        else:
            print("Threshold not reached")
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    T = len(rewards_arr)
    window = max(1, window)

    for t in range(window - 1, T):
        segment = rewards_arr[t - window + 1 : t + 1]
        if segment.mean() >= threshold:
            return t
    return None


# ---------------------------------------------------------------------------
# Statistical Significance Test (Welch's t-test)
# ---------------------------------------------------------------------------

def statistical_significance_test(
    rewards_a: list[float] | np.ndarray,
    rewards_b: list[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, float | bool | str]:
    """Test whether two sets of episode rewards differ significantly.

    Uses Welch's t-test (unequal variances), which is more robust than
    Student's t-test when sample sizes or variances differ.

    Args:
        rewards_a: Episode rewards for agent / configuration A.
        rewards_b: Episode rewards for agent / configuration B.
        alpha: Significance level (default ``0.05``).
        alternative: One of ``"two-sided"``, ``"less"``, ``"greater"``.
            Passed directly to ``scipy.stats.ttest_ind``.

    Returns:
        Dictionary with:
        ``t_statistic``, ``p_value``, ``significant`` (bool),
        ``alpha``, ``alternative``,
        ``mean_a``, ``std_a``, ``n_a``,
        ``mean_b``, ``std_b``, ``n_b``,
        ``interpretation`` (human-readable string).

    Example::

        result = statistical_significance_test(dqn_rewards, ppo_rewards)
        print(result["interpretation"])
        # "PPO is significantly better than DQN (p=0.0032)"
    """
    from scipy import stats  # imported here so scipy is only required for this fn

    a = np.asarray(rewards_a, dtype=np.float64)
    b = np.asarray(rewards_b, dtype=np.float64)

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False, alternative=alternative)
    significant = bool(p_value < alpha)

    mean_a, mean_b = float(a.mean()), float(b.mean())
    if significant:
        winner = "A" if mean_a > mean_b else "B"
        interp = f"Agent {winner} is significantly different (p={p_value:.4f}, alpha={alpha})"
    else:
        interp = f"No significant difference detected (p={p_value:.4f}, alpha={alpha})"

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": significant,
        "alpha": alpha,
        "alternative": alternative,
        "mean_a": mean_a,
        "std_a": float(a.std()),
        "n_a": len(a),
        "mean_b": mean_b,
        "std_b": float(b.std()),
        "n_b": len(b),
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# ASCII Policy Table for Gridworlds
# ---------------------------------------------------------------------------

_ACTION_SYMBOLS: dict[int, str] = {
    0: "^",   # up
    1: "v",   # down
    2: "<",   # left
    3: ">",   # right
}


def render_policy_table(
    q_values: np.ndarray,
    env_shape: tuple[int, int],
    *,
    wall_states: set[int] | None = None,
    goal_states: set[int] | None = None,
    action_symbols: dict[int, str] | None = None,
    cell_width: int = 5,
) -> str:
    """Render a greedy policy as an ASCII table for a gridworld environment.

    Displays the best action (argmax over Q-values) at each cell. Walls and
    goal states receive special markers instead of action arrows.

    State numbering follows row-major order: state ``s = row * n_cols + col``.

    Args:
        q_values: Array of Q-values, shape ``(n_states, n_actions)``.
        env_shape: ``(n_rows, n_cols)`` of the gridworld.
        wall_states: Set of state indices treated as impassable walls (shown as
            ``"#"``).
        goal_states: Set of state indices treated as terminal goals (shown as
            ``"G"``).
        action_symbols: Mapping from action index to display string. Defaults
            to ``{0:"^", 1:"v", 2:"<", 3:">"}``.
        cell_width: Character width of each cell (minimum 3).

    Returns:
        Multi-line string. Print it directly or write to a file.

    Example::

        # 4x4 gridworld, 4 actions (up/down/left/right)
        q = np.random.rand(16, 4)
        print(render_policy_table(q, (4, 4), goal_states={15}, wall_states={5}))

        #  ----+----+----+----
        #  >   | >  | v  | v
        #  ----+----+----+----
        #  ^   | #  | v  | v
        #  ...
    """
    symbols = action_symbols if action_symbols is not None else _ACTION_SYMBOLS
    walls = wall_states or set()
    goals = goal_states or set()

    n_rows, n_cols = env_shape
    n_states = n_rows * n_cols
    if q_values.shape[0] != n_states:
        raise ValueError(
            f"q_values has {q_values.shape[0]} states but env_shape implies "
            f"{n_states} states."
        )

    cell_width = max(3, cell_width)
    divider = ("+" + "-" * cell_width) * n_cols + "+"

    lines: list[str] = [divider]
    for row in range(n_rows):
        row_cells: list[str] = []
        for col in range(n_cols):
            state = row * n_cols + col
            if state in walls:
                marker = "#"
            elif state in goals:
                marker = "G"
            else:
                best_action = int(np.argmax(q_values[state]))
                marker = symbols.get(best_action, str(best_action))
            row_cells.append(marker.center(cell_width))
        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(divider)

    return "\n".join(lines)

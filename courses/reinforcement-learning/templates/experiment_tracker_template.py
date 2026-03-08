"""
Experiment Tracker Template — Lightweight JSON-based logging for RL experiments.
Works with: Any agent/environment pair; no external tracking service required
Time to working: 5 minutes

Features:
- JSON episode log with hyperparameters, rewards, lengths, and custom metrics
- Agent checkpoint save/load with episode metadata
- Matplotlib plots: reward curve (with rolling average), episode length, custom metrics
- Comparison mode: overlay multiple runs on the same axes

Dependencies: numpy, matplotlib (stdlib + these two only; no MLflow, W&B, etc.)

Example use cases:
- Reproducing experiments with full hyperparameter logs
- Comparing epsilon-greedy vs Thompson-Sampling on the same environment
- Monitoring training stability without spinning up an external service
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend; switch to "TkAgg" for live display

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ROLLING_WINDOW: int = 50        # Episodes for rolling-average window
DEFAULT_CHECKPOINT_SUBDIR: str = "checkpoints"
LOG_FILENAME: str = "experiment_log.json"
FIGURE_DPI: int = 120


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class EpisodeRecord:
    """One episode's worth of logged data.

    Attributes:
        episode:     Episode number (1-indexed).
        reward:      Total undiscounted reward for this episode.
        length:      Number of environment steps taken.
        timestamp:   Unix timestamp at episode end.
        metrics:     Optional dict of custom scalar metrics.
    """

    episode: int
    reward: float
    length: int
    timestamp: float = field(default_factory=time.time)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CheckpointMeta:
    """Metadata stored alongside each agent checkpoint.

    Attributes:
        episode:       Episode number at which the checkpoint was saved.
        mean_reward:   Rolling mean reward at save time (last N episodes).
        agent_class:   Fully qualified class name of the agent.
        timestamp:     Unix timestamp at save time.
        extra:         Any additional key/value data the caller wants to keep.
    """

    episode: int
    mean_reward: float
    agent_class: str
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EXPERIMENT TRACKER
# ============================================================================


class ExperimentTracker:
    """Lightweight experiment logger with checkpointing and plotting.

    All data is written to a single JSON file inside ``experiment_dir``.
    Agent checkpoints are saved in a ``checkpoints/`` subdirectory.  No
    external services or heavy dependencies are required.

    Args:
        experiment_dir:  Root directory for this experiment's outputs.
                         Created automatically if it does not exist.
        experiment_name: Human-readable name used in plot titles.
        rolling_window:  Number of episodes for rolling-average computation.

    Example::

        tracker = ExperimentTracker("./runs/dqn_cartpole", "DQN CartPole")
        tracker.log_hyperparams({"lr": 1e-3, "gamma": 0.99, "batch_size": 64})

        for episode in range(300):
            # ... training code ...
            tracker.log_episode(reward=ep_reward, length=ep_length)
            if episode % 50 == 0:
                tracker.save_checkpoint(agent, episode)

        tracker.plot_training_curves(save_path="./runs/dqn_cartpole/curves.png")
        print(tracker.summary())
    """

    def __init__(
        self,
        experiment_dir: str | os.PathLike,
        experiment_name: str = "experiment",
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.rolling_window = rolling_window

        self._log_path: Path = self.experiment_dir / LOG_FILENAME
        self._checkpoint_dir: Path = self.experiment_dir / DEFAULT_CHECKPOINT_SUBDIR
        self._checkpoint_dir.mkdir(exist_ok=True)

        # In-memory state
        self._hyperparams: Dict[str, Any] = {}
        self._episodes: List[EpisodeRecord] = []
        self._checkpoints: List[CheckpointMeta] = []
        self._start_time: float = time.time()

        # Load existing log if resuming
        if self._log_path.exists():
            self._load_log()

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Record experiment hyperparameters (overrides any existing values).

        Call once before training begins.  All values must be JSON-serialisable.

        Args:
            params: Dictionary of hyperparameter names to values.

        Example::

            tracker.log_hyperparams({
                "lr": 1e-3,
                "gamma": 0.99,
                "batch_size": 64,
                "hidden_dim": 128,
            })
        """
        self._hyperparams.update(params)
        self._persist()

    def log_episode(
        self,
        reward: float,
        length: int,
        info: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record statistics for one completed episode.

        Args:
            reward: Total undiscounted episode reward.
            length: Number of steps taken in the episode.
            info:   Optional dict of custom scalar metrics to log
                    (e.g. ``{"constraint_violations": 2, "goal_reached": 1.0}``).

        Example::

            tracker.log_episode(reward=ep_reward, length=ep_length)
            # With custom metrics:
            tracker.log_episode(reward=ep_reward, length=ep_length,
                                info={"constraint_violations": 3})
        """
        episode_num = len(self._episodes) + 1
        record = EpisodeRecord(
            episode=episode_num,
            reward=float(reward),
            length=int(length),
            metrics=info or {},
        )
        self._episodes.append(record)
        self._persist()

    def save_checkpoint(self, agent: Any, episode: int, **extra: Any) -> Path:
        """Persist the agent and record checkpoint metadata.

        The agent's ``save(path)`` method is called with the checkpoint path.
        The method must accept a path-like argument (see ``rl_agent_template.py``).

        Args:
            agent:   Any agent that implements a ``save(path)`` method.
            episode: Current episode number (used in the filename).
            **extra: Additional key/value pairs to include in metadata.

        Returns:
            Path to the saved checkpoint file (without extension).

        Example::

            path = tracker.save_checkpoint(agent, episode=100)
            print(f"Saved to {path}")
        """
        ckpt_name = f"ep{episode:06d}"
        ckpt_path = self._checkpoint_dir / ckpt_name
        agent.save(ckpt_path)

        recent_rewards = [ep.reward for ep in self._episodes[-self.rolling_window:]]
        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0

        meta = CheckpointMeta(
            episode=episode,
            mean_reward=mean_reward,
            agent_class=type(agent).__qualname__,
            extra=extra,
        )
        self._checkpoints.append(meta)
        self._persist()
        return ckpt_path

    def load_checkpoint(self, agent: Any, path: str | os.PathLike) -> None:
        """Load a checkpoint into an agent.

        Args:
            agent: Agent instance whose ``load(path)`` method will be called.
            path:  Path to the checkpoint (with or without extension).

        Example::

            tracker.load_checkpoint(agent, "./runs/dqn_cartpole/checkpoints/ep000100")
        """
        agent.load(path)

    def load_best_checkpoint(self, agent: Any) -> Optional[Path]:
        """Load the checkpoint with the highest recorded mean reward.

        Args:
            agent: Agent instance whose ``load(path)`` method will be called.

        Returns:
            Path to the loaded checkpoint, or None if no checkpoints exist.
        """
        if not self._checkpoints:
            return None
        best = max(self._checkpoints, key=lambda m: m.mean_reward)
        ckpt_name = f"ep{best.episode:06d}"
        ckpt_path = self._checkpoint_dir / ckpt_name
        agent.load(ckpt_path)
        return ckpt_path

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_training_curves(
        self,
        save_path: Optional[str | os.PathLike] = None,
        show: bool = False,
        extra_metrics: Optional[List[str]] = None,
    ) -> plt.Figure:
        """Generate a multi-panel training curve figure.

        Panels produced:
        1. Episode reward (raw + rolling average)
        2. Episode length (raw + rolling average)
        3. One panel per metric key in ``extra_metrics`` (if provided and logged)

        Args:
            save_path:     If set, save the figure to this path (PNG/PDF/SVG).
                           Defaults to ``<experiment_dir>/training_curves.png``.
            show:          If True, call ``plt.show()`` after rendering.
            extra_metrics: List of custom metric keys to plot in additional panels.

        Returns:
            The matplotlib ``Figure`` object.

        Example::

            tracker.plot_training_curves(
                save_path="./curves.png",
                extra_metrics=["constraint_violations"],
            )
        """
        if not self._episodes:
            raise RuntimeError("No episodes logged yet. Run training before plotting.")

        extra_metrics = extra_metrics or []
        # Only plot metrics that were actually logged
        available_metrics = [
            m for m in extra_metrics
            if any(m in ep.metrics for ep in self._episodes)
        ]
        n_panels = 2 + len(available_metrics)

        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), dpi=FIGURE_DPI)
        if n_panels == 1:
            axes = [axes]

        episodes = [ep.episode for ep in self._episodes]
        rewards = [ep.reward for ep in self._episodes]
        lengths = [ep.length for ep in self._episodes]

        # Panel 1: Rewards
        self._plot_metric_panel(
            ax=axes[0],
            episodes=episodes,
            values=rewards,
            label="Episode Reward",
            color="#2196F3",
            rolling_window=self.rolling_window,
            title=f"{self.experiment_name} — Reward",
        )

        # Panel 2: Episode lengths
        self._plot_metric_panel(
            ax=axes[1],
            episodes=episodes,
            values=lengths,
            label="Episode Length",
            color="#4CAF50",
            rolling_window=self.rolling_window,
            title="Episode Length",
        )

        # Extra metric panels
        for ax, key in zip(axes[2:], available_metrics):
            values = [ep.metrics.get(key, float("nan")) for ep in self._episodes]
            self._plot_metric_panel(
                ax=ax,
                episodes=episodes,
                values=values,
                label=key.replace("_", " ").title(),
                color="#FF5722",
                rolling_window=self.rolling_window,
                title=key.replace("_", " ").title(),
            )

        fig.tight_layout()

        dest = Path(save_path) if save_path else self.experiment_dir / "training_curves.png"
        fig.savefig(dest, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    @staticmethod
    def compare_runs(
        trackers: List["ExperimentTracker"],
        metric: str = "reward",
        save_path: Optional[str | os.PathLike] = None,
        show: bool = False,
    ) -> plt.Figure:
        """Overlay rolling-average reward curves from multiple experiment runs.

        Args:
            trackers:  List of ``ExperimentTracker`` instances to compare.
            metric:    Which metric to compare: ``"reward"`` or ``"length"``.
            save_path: File path to save the figure; defaults to CWD.
            show:      If True, call ``plt.show()`` after rendering.

        Returns:
            The matplotlib ``Figure`` object.

        Example::

            t1 = ExperimentTracker("./runs/dqn_run1", "DQN run-1")
            t2 = ExperimentTracker("./runs/dqn_run2", "DQN run-2")
            ExperimentTracker.compare_runs([t1, t2], metric="reward",
                                           save_path="./comparison.png")
        """
        fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)
        palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for idx, tracker in enumerate(trackers):
            if not tracker._episodes:
                continue
            episodes = [ep.episode for ep in tracker._episodes]
            if metric == "reward":
                values = [ep.reward for ep in tracker._episodes]
                ylabel = "Episode Reward"
            elif metric == "length":
                values = [ep.length for ep in tracker._episodes]
                ylabel = "Episode Length"
            else:
                values = [ep.metrics.get(metric, float("nan")) for ep in tracker._episodes]
                ylabel = metric.replace("_", " ").title()

            color = palette[idx % len(palette)]
            rolling = _rolling_mean(values, tracker.rolling_window)

            ax.plot(episodes, values, alpha=0.2, color=color, linewidth=0.8)
            ax.plot(
                episodes[len(episodes) - len(rolling):],
                rolling,
                color=color,
                linewidth=2.0,
                label=tracker.experiment_name,
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Run Comparison — {ylabel}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        dest = Path(save_path) if save_path else Path("comparison.png")
        fig.savefig(dest, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable experiment summary.

        Returns:
            Multi-line string covering hyperparameters, reward statistics,
            best checkpoint, and total training time.
        """
        if not self._episodes:
            return f"[{self.experiment_name}] No episodes logged."

        rewards = [ep.reward for ep in self._episodes]
        lengths = [ep.length for ep in self._episodes]
        elapsed = time.time() - self._start_time

        best_ckpt = (
            max(self._checkpoints, key=lambda m: m.mean_reward)
            if self._checkpoints
            else None
        )

        lines = [
            "=" * 60,
            f"Experiment : {self.experiment_name}",
            f"Directory  : {self.experiment_dir}",
            f"Episodes   : {len(rewards)}",
            f"Elapsed    : {elapsed / 60:.1f} min",
            "-" * 60,
            "Reward statistics:",
            f"  mean  : {np.mean(rewards):10.3f}",
            f"  std   : {np.std(rewards):10.3f}",
            f"  min   : {np.min(rewards):10.3f}",
            f"  max   : {np.max(rewards):10.3f}",
            f"  last {self.rolling_window}-ep avg : "
            f"{np.mean(rewards[-self.rolling_window:]):10.3f}",
            "-" * 60,
            "Length statistics:",
            f"  mean  : {np.mean(lengths):10.1f}",
            f"  max   : {np.max(lengths):10.0f}",
        ]

        if self._hyperparams:
            lines += ["-" * 60, "Hyperparameters:"]
            for k, v in self._hyperparams.items():
                lines.append(f"  {k:<20}: {v}")

        if best_ckpt:
            lines += [
                "-" * 60,
                f"Best checkpoint: episode {best_ckpt.episode} "
                f"(mean_reward={best_ckpt.mean_reward:.3f})",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence (private)
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Atomically write the full experiment log to JSON."""
        payload = {
            "experiment_name": self.experiment_name,
            "hyperparams": self._hyperparams,
            "episodes": [asdict(ep) for ep in self._episodes],
            "checkpoints": [asdict(m) for m in self._checkpoints],
        }
        tmp_path = self._log_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        # Atomic rename — avoids a partial write corrupting the log
        tmp_path.replace(self._log_path)

    def _load_log(self) -> None:
        """Restore in-memory state from an existing JSON log file."""
        with open(self._log_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self.experiment_name = payload.get("experiment_name", self.experiment_name)
        self._hyperparams = payload.get("hyperparams", {})

        self._episodes = [
            EpisodeRecord(**ep) for ep in payload.get("episodes", [])
        ]
        self._checkpoints = [
            CheckpointMeta(**m) for m in payload.get("checkpoints", [])
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_metric_panel(
        ax: plt.Axes,
        episodes: List[int],
        values: List[float],
        label: str,
        color: str,
        rolling_window: int,
        title: str,
    ) -> None:
        """Render a single time-series panel with raw data and rolling average.

        Args:
            ax:             Matplotlib axes to draw on.
            episodes:       X-axis values (episode numbers).
            values:         Y-axis values (metric per episode).
            label:          Legend label for the rolling average line.
            color:          Hex colour for both raw and rolling lines.
            rolling_window: Window size for the rolling mean.
            title:          Panel title.
        """
        rolling = _rolling_mean(values, rolling_window)
        offset = len(episodes) - len(rolling)

        ax.plot(episodes, values, alpha=0.25, color=color, linewidth=0.8)
        ax.plot(
            episodes[offset:],
            rolling,
            color=color,
            linewidth=2.0,
            label=f"{label} ({rolling_window}-ep avg)",
        )
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)


# ============================================================================
# MODULE-LEVEL HELPER
# ============================================================================


def _rolling_mean(values: List[float], window: int) -> List[float]:
    """Compute a simple rolling mean using a uniform convolution.

    Args:
        values: Input sequence of floats.
        window: Window width.

    Returns:
        List of rolling-mean values (length = max(0, len(values) - window + 1)).
    """
    if len(values) < window:
        return []
    arr = np.array(values, dtype=np.float64)
    kernel = np.ones(window) / window
    # "valid" mode returns only fully-overlapping windows
    return list(np.convolve(arr, kernel, mode="valid"))


# ============================================================================
# RUNNABLE DEMO
# ============================================================================


if __name__ == "__main__":
    import tempfile
    import random

    print("=== ExperimentTracker Demo ===\n")

    # ------------------------------------------------------------------
    # Simulate two training runs with synthetic data
    # ------------------------------------------------------------------
    def _simulate_run(
        name: str,
        base_dir: Path,
        n_episodes: int,
        noise_scale: float,
    ) -> ExperimentTracker:
        """Create a tracker and fill it with simulated episode data."""
        tracker = ExperimentTracker(base_dir / name, experiment_name=name)
        tracker.log_hyperparams({
            "lr": 1e-3,
            "gamma": 0.99,
            "batch_size": 64,
            "noise_scale": noise_scale,
        })

        for ep in range(1, n_episodes + 1):
            # Synthetic reward: slowly rising trend + noise
            base_reward = min(200.0, 10.0 + ep * 0.5)
            reward = base_reward + random.gauss(0, noise_scale)
            length = max(1, int(base_reward + random.gauss(0, noise_scale * 0.5)))
            constraint_violations = max(0, int(random.gauss(5 - ep * 0.01, 2)))

            tracker.log_episode(
                reward=reward,
                length=length,
                info={"constraint_violations": float(constraint_violations)},
            )

        return tracker

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)

        tracker_a = _simulate_run("DQN_lr1e-3", run_dir, n_episodes=300, noise_scale=15.0)
        tracker_b = _simulate_run("DQN_lr5e-4", run_dir, n_episodes=300, noise_scale=10.0)

        # Print summaries
        print(tracker_a.summary())
        print()
        print(tracker_b.summary())

        # Plot individual curves
        fig_a = tracker_a.plot_training_curves(
            save_path=run_dir / "curves_a.png",
            extra_metrics=["constraint_violations"],
        )
        print(f"\nTraining curves saved to {run_dir / 'curves_a.png'}")

        # Overlay comparison
        fig_cmp = ExperimentTracker.compare_runs(
            [tracker_a, tracker_b],
            metric="reward",
            save_path=run_dir / "comparison.png",
        )
        print(f"Comparison plot saved to {run_dir / 'comparison.png'}")

        # Demonstrate checkpoint save/load (mock agent with save/load)
        class _MockAgent:
            """Minimal mock agent for the checkpoint demo."""

            def save(self, path: os.PathLike) -> None:
                dest = Path(path).with_suffix(".json")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "w") as fh:
                    json.dump({"weights": [1.0, 2.0, 3.0]}, fh)

            def load(self, path: os.PathLike) -> None:
                src = Path(path).with_suffix(".json")
                with open(src) as fh:
                    data = json.load(fh)
                print(f"  Loaded weights: {data['weights']}")

        agent = _MockAgent()
        print("\nSaving checkpoint at episode 100 ...")
        ckpt = tracker_a.save_checkpoint(agent, episode=100, notes="mid-training snapshot")
        print(f"  Checkpoint path: {ckpt}")

        print("Loading best checkpoint ...")
        best_path = tracker_a.load_best_checkpoint(agent)
        print(f"  Best checkpoint: {best_path}")

    print("\nDemo complete.")

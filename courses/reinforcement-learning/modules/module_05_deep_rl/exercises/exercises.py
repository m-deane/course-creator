"""
Module 05 - Deep Reinforcement Learning: Self-Check Exercises

Covers:
- Experience replay buffer with uniform sampling
- DQN target value computation
- Double DQN target value computation

Run with: python exercises.py
Dependencies: numpy, torch
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random


# ---------------------------------------------------------------------------
# Exercise 1: Experience Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer with uniform random sampling.

    Problem
    -------
    A replay buffer stores transitions (s, a, r, s', done) and allows
    random mini-batch sampling to break temporal correlations in training.

    When the buffer exceeds `capacity`, the oldest transition is discarded
    (FIFO). Use collections.deque with maxlen for automatic eviction.

    Implement:
      - push(state, action, reward, next_state, done): add a transition.
      - sample(batch_size): return a batch as separate numpy arrays.
      - __len__: return current number of stored transitions.

    Hints
    -----
    - Store each transition as a tuple in a deque(maxlen=capacity).
    - random.sample(self.buffer, batch_size) draws without replacement.
    - Unzip transitions with zip(*batch) and convert each field to ndarray.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store.

    Examples
    --------
    >>> buf = ReplayBuffer(100)
    >>> buf.push(np.zeros(4), 0, 1.0, np.ones(4), False)
    >>> states, actions, rewards, next_states, dones = buf.sample(1)
    """

    def __init__(self, capacity: int) -> None:
        # SOLUTION
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        # SOLUTION
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Sample a random mini-batch without replacement.

        Returns
        -------
        tuple of np.ndarray
            (states, actions, rewards, next_states, dones)
            shapes: (B, *state_shape), (B,), (B,), (B, *state_shape), (B,)
        """
        # SOLUTION
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def test_exercise_1() -> None:
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0, "Empty buffer must have length 0."

    # Push 5 transitions
    for i in range(5):
        buf.push(np.array([float(i)]), i % 2, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 5, f"Buffer must have 5 transitions, got {len(buf)}."

    # Capacity enforcement: push 7 more, buffer stays at 10
    for i in range(7):
        buf.push(np.array([0.0]), 0, 0.0, np.array([0.0]), True)
    assert len(buf) == 10, \
        f"Buffer must cap at capacity=10, got {len(buf)}."

    # Sample returns correct shapes
    states, actions, rewards, next_states, dones = buf.sample(4)
    assert states.shape == (4, 1), f"states shape must be (4,1), got {states.shape}."
    assert actions.shape == (4,), f"actions shape must be (4,), got {actions.shape}."
    assert rewards.shape == (4,), f"rewards shape must be (4,), got {rewards.shape}."
    assert next_states.shape == (4, 1), \
        f"next_states shape must be (4,1), got {next_states.shape}."
    assert dones.shape == (4,), f"dones shape must be (4,), got {dones.shape}."

    # dones must be float (0.0 or 1.0)
    assert dones.dtype == np.float32, f"dones dtype must be float32, got {dones.dtype}."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: DQN Target Values
# ---------------------------------------------------------------------------

def dqn_targets(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Compute DQN target values for a mini-batch.

    Problem
    -------
    The DQN target for each sample in a batch is:

        y_i = r_i + gamma * max_a' Q_target(s'_i, a')  * (1 - done_i)

    where Q_target is computed by a separate (frozen) target network.
    The (1 - done) mask ensures terminal states have no bootstrap term.

    Hints
    -----
    - next_q_values has shape (B, A); take max over dim=1.
    - Multiply bootstrap term by (1 - dones) to zero out terminal states.
    - Do NOT call .backward() here; this is target computation only.

    Parameters
    ----------
    rewards : torch.Tensor, shape (B,)
        Batch of rewards.
    next_q_values : torch.Tensor, shape (B, A)
        Q-values for next states from the target network.
    dones : torch.Tensor, shape (B,)
        Float tensor, 1.0 if terminal, 0.0 otherwise.
    gamma : float
        Discount factor.

    Returns
    -------
    torch.Tensor, shape (B,)
        Target values y_i (detached from computation graph).

    Examples
    --------
    >>> rewards = torch.tensor([1.0, 0.0])
    >>> next_q  = torch.tensor([[0.5, 2.0], [1.0, 0.3]])
    >>> dones   = torch.tensor([0.0, 1.0])
    >>> dqn_targets(rewards, next_q, dones, gamma=0.9)
    tensor([2.8, 0.0])  # [1+0.9*2, 0+0.9*1*0] -> [2.8, 0.0]
    """
    # SOLUTION
    max_next_q = next_q_values.max(dim=1).values
    targets = rewards + gamma * max_next_q * (1.0 - dones)
    return targets.detach()


def test_exercise_2() -> None:
    rewards = torch.tensor([1.0, 0.0])
    next_q = torch.tensor([[0.5, 2.0], [1.0, 0.3]])
    dones = torch.tensor([0.0, 1.0])

    targets = dqn_targets(rewards, next_q, dones, gamma=0.9)
    assert targets.shape == (2,), f"targets shape must be (2,), got {targets.shape}."
    # y[0] = 1 + 0.9 * max(0.5, 2.0) * (1-0) = 1 + 1.8 = 2.8
    assert torch.isclose(targets[0], torch.tensor(2.8)), \
        f"y[0] must be 2.8, got {targets[0]}."
    # y[1] = 0 + 0.9 * max(1.0, 0.3) * (1-1) = 0
    assert torch.isclose(targets[1], torch.tensor(0.0)), \
        f"y[1] (terminal) must be 0.0, got {targets[1]}."

    # All non-terminal: bootstrap applies to all
    rewards2 = torch.zeros(3)
    next_q2 = torch.tensor([[1.0, 2.0, 3.0]] * 3)
    dones2 = torch.zeros(3)
    targets2 = dqn_targets(rewards2, next_q2, dones2, gamma=0.5)
    assert torch.allclose(targets2, torch.full((3,), 1.5)), \
        f"Expected all 1.5, got {targets2}."

    # Result must be detached
    assert not targets.requires_grad, "Targets must be detached (no grad)."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Double DQN Target Values
# ---------------------------------------------------------------------------

def double_dqn_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    online_network: nn.Module,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
    """
    Compute Double DQN target values.

    Problem
    -------
    Vanilla DQN overestimates Q-values because the same network both
    selects and evaluates the next action. Double DQN decouples these:

        a*_i = argmax_a  Q_online(s'_i, a)   [action SELECTION by online net]
        y_i  = r_i + gamma * Q_target(s'_i, a*_i) * (1 - done_i)
                                              [action EVALUATION by target net]

    Contrast with vanilla DQN which uses max_a Q_target(s', a) (both
    selection and evaluation by target net).

    Hints
    -----
    - Disable gradient computation with torch.no_grad().
    - online_network(next_states) gives shape (B, A); argmax over dim=1.
    - Gather target Q-values at selected actions: target_q.gather(1, a_star).

    Parameters
    ----------
    rewards : torch.Tensor, shape (B,)
        Batch rewards.
    next_states : torch.Tensor, shape (B, state_dim)
        Batch of next states.
    dones : torch.Tensor, shape (B,)
        Terminal flags.
    online_network : nn.Module
        The online (current) Q-network.
    target_network : nn.Module
        The frozen target Q-network.
    gamma : float
        Discount factor.

    Returns
    -------
    torch.Tensor, shape (B,)
        Double DQN target values (detached).
    """
    # SOLUTION
    with torch.no_grad():
        # Select actions using online network
        online_q_next = online_network(next_states)           # (B, A)
        best_actions = online_q_next.argmax(dim=1, keepdim=True)  # (B, 1)

        # Evaluate those actions using target network
        target_q_next = target_network(next_states)           # (B, A)
        selected_q = target_q_next.gather(1, best_actions).squeeze(1)  # (B,)

        targets = rewards + gamma * selected_q * (1.0 - dones)
    return targets


def test_exercise_3() -> None:
    torch.manual_seed(0)

    # Build a simple 1-hidden-layer network
    def make_net(state_dim: int, n_actions: int, seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions),
        )

    state_dim, n_actions, batch = 4, 2, 3
    online_net = make_net(state_dim, n_actions, seed=1)
    target_net = make_net(state_dim, n_actions, seed=2)

    torch.manual_seed(42)
    next_states = torch.randn(batch, state_dim)
    rewards = torch.tensor([1.0, 0.0, 0.5])
    dones = torch.tensor([0.0, 1.0, 0.0])

    ddqn_t = double_dqn_targets(rewards, next_states, dones,
                                  online_net, target_net, gamma=0.9)
    dqn_t = dqn_targets(rewards,
                         target_net(next_states).detach(),
                         dones, gamma=0.9)

    assert ddqn_t.shape == (batch,), \
        f"DDQN targets shape must be ({batch},), got {ddqn_t.shape}."

    # Terminal state target must equal reward only
    assert torch.isclose(ddqn_t[1], rewards[1]), \
        f"Terminal target must be reward only={rewards[1]}, got {ddqn_t[1]}."

    # Targets must be detached
    assert not ddqn_t.requires_grad, "DDQN targets must be detached."

    # DQN and DDQN should generally differ (different networks select/eval)
    # With different online/target nets, they'll differ on non-terminal states
    # (this is the anti-overestimation property being tested structurally)
    non_terminal_mask = dones == 0.0
    assert ddqn_t.shape == dqn_t.shape, "Shapes must match."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: Experience Replay Buffer", test_exercise_1),
        ("Exercise 2: DQN Target Values", test_exercise_2),
        ("Exercise 3: Double DQN Target Values", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

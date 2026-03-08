"""
Module 06 - Policy Gradient Methods: Self-Check Exercises

Covers:
- REINFORCE loss computation from log-probabilities and returns
- Discounted return computation with baseline subtraction
- Generalized Advantage Estimation (GAE)

Run with: python exercises.py
Dependencies: numpy, torch
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Exercise 1: REINFORCE Loss
# ---------------------------------------------------------------------------

def reinforce_loss(
    log_probs: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the REINFORCE policy gradient loss.

    Problem
    -------
    The REINFORCE objective (to maximise) is:

        J(theta) = E[sum_t G_t * log pi_theta(a_t | s_t)]

    To use gradient *descent* optimisers, we return the *negative* of this:

        loss = -mean( log_pi(a_t | s_t) * G_t )   for t = 0..T-1

    Hints
    -----
    - Element-wise multiply log_probs and returns, then take the mean.
    - Negate the result (we minimise the loss to maximise the objective).
    - Do NOT detach returns — they are treated as fixed weights here, but
      the loss must remain connected to the policy parameters via log_probs.

    Parameters
    ----------
    log_probs : torch.Tensor, shape (T,)
        Log-probabilities of the selected actions at each timestep.
    returns : torch.Tensor, shape (T,)
        Discounted returns G_t for each timestep.

    Returns
    -------
    torch.Tensor
        Scalar loss value (negative policy gradient objective).

    Examples
    --------
    >>> log_probs = torch.tensor([-1.0, -0.5])
    >>> returns   = torch.tensor([2.0, 1.0])
    >>> reinforce_loss(log_probs, returns)
    tensor(1.25)  # -mean([-2.0, -0.5]) = -(-1.25) = 1.25
    """
    # SOLUTION
    return -(log_probs * returns).mean()


def test_exercise_1() -> None:
    log_probs = torch.tensor([-1.0, -0.5])
    returns = torch.tensor([2.0, 1.0])

    loss = reinforce_loss(log_probs, returns)
    # -mean([-1*2, -0.5*1]) = -mean([-2, -0.5]) = -(-1.25) = 1.25
    assert loss.shape == torch.Size([]), f"Loss must be scalar, got shape {loss.shape}."
    assert torch.isclose(loss, torch.tensor(1.25)), \
        f"Expected 1.25, got {loss.item()}."

    # All positive returns with negative log-probs: loss must be positive
    lp = torch.tensor([-0.1, -0.2, -0.3])
    ret = torch.tensor([5.0, 4.0, 3.0])
    loss2 = reinforce_loss(lp, ret)
    assert loss2.item() > 0, "Loss with positive returns and negative log-probs must be > 0."

    # Zero returns: loss is 0
    loss3 = reinforce_loss(torch.zeros(5), torch.zeros(5))
    assert torch.isclose(loss3, torch.tensor(0.0)), \
        f"Zero-return loss must be 0, got {loss3}."

    # Loss is connected to graph (has grad_fn)
    lp_param = torch.tensor([-1.0, -0.5], requires_grad=True)
    loss4 = reinforce_loss(lp_param, torch.tensor([2.0, 1.0]))
    assert loss4.grad_fn is not None, "Loss must retain gradient graph."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Discounted Returns with Baseline Subtraction
# ---------------------------------------------------------------------------

def compute_returns_with_baseline(
    rewards: list[float],
    gamma: float,
    baseline: float = 0.0,
) -> torch.Tensor:
    """
    Compute discounted returns and subtract a baseline to reduce variance.

    Problem
    -------
    For a trajectory of length T:
      1. Compute discounted return G_t = sum_{k=t}^{T-1} gamma^{k-t} * r_k.
      2. Subtract a baseline b: A_t = G_t - b.

    A common baseline is the mean return over the episode, which centres
    the advantages around zero and reduces gradient variance without
    introducing bias (since E[b * grad log pi] = b * E[grad log pi] = 0
    for a state-independent baseline).

    Hints
    -----
    - Compute returns via reverse accumulation (O(T) time).
    - If baseline=None or not provided, use mean(G_0..G_{T-1}).
    - Return as a torch.Tensor of dtype float32.

    Parameters
    ----------
    rewards : list of float
        Trajectory rewards [r_0, r_1, ..., r_{T-1}].
    gamma : float
        Discount factor.
    baseline : float or None
        Value to subtract. If None, use the mean return.

    Returns
    -------
    torch.Tensor, shape (T,)
        Baseline-adjusted returns A_t = G_t - baseline.

    Examples
    --------
    >>> compute_returns_with_baseline([1.0, 1.0], gamma=1.0, baseline=0.0)
    tensor([2., 1.])  # G_0=2, G_1=1; A_t = G_t - 0
    """
    # SOLUTION
    T = len(rewards)
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns = list(reversed(returns))  # G_0, G_1, ..., G_{T-1}

    G_tensor = torch.tensor(returns, dtype=torch.float32)
    b = baseline if baseline is not None else G_tensor.mean().item()
    return G_tensor - b


def test_exercise_2() -> None:
    # gamma=1, baseline=0: just the raw returns
    A = compute_returns_with_baseline([1.0, 1.0], gamma=1.0, baseline=0.0)
    assert A.shape == (2,), f"Shape must be (2,), got {A.shape}."
    assert torch.allclose(A, torch.tensor([2.0, 1.0])), \
        f"Expected [2.0, 1.0], got {A}."

    # Mean baseline: advantages sum to 0 (centred)
    A_mean = compute_returns_with_baseline([1.0, 2.0, 3.0], gamma=1.0, baseline=None)
    assert torch.isclose(A_mean.mean(), torch.tensor(0.0), atol=1e-6), \
        f"Mean-baseline advantages must sum to ~0, got mean={A_mean.mean()}."

    # gamma=0: each return equals its own reward
    A_g0 = compute_returns_with_baseline([4.0, 2.0, 1.0], gamma=0.0, baseline=0.0)
    assert torch.allclose(A_g0, torch.tensor([4.0, 2.0, 1.0])), \
        f"gamma=0 returns must equal rewards, got {A_g0}."

    # Output dtype
    assert A.dtype == torch.float32, f"Output dtype must be float32, got {A.dtype}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Generalized Advantage Estimation (GAE)
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
    last_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimation (GAE-lambda).

    Problem
    -------
    GAE provides a bias-variance trade-off for advantage estimation:

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   (TD residual)
        A_t^GAE = sum_{l=0}^{T-1-t} (gamma * lam)^l * delta_{t+l}

    Equivalently, compute backwards:
        A_T = 0
        A_t = delta_t + gamma * lam * A_{t+1}

    Special cases:
      - lam=0: A_t = delta_t (TD(0) advantage, low variance, high bias)
      - lam=1: A_t = sum of discounted TD residuals = Monte Carlo advantage

    Hints
    -----
    - Append last_value to values for the bootstrap at the final step.
    - Iterate rewards in reverse order, accumulating advantage.
    - Return a torch.Tensor of dtype float32.

    Parameters
    ----------
    rewards : list of float
        Trajectory rewards [r_0, ..., r_{T-1}].
    values : list of float
        Value estimates [V(s_0), ..., V(s_{T-1})].
    gamma : float
        Discount factor.
    lam : float
        GAE lambda in [0, 1].
    last_value : float
        Bootstrap value V(s_T) (0 if terminal).

    Returns
    -------
    torch.Tensor, shape (T,)
        GAE advantages A_t^GAE for t = 0..T-1.

    Examples
    --------
    >>> compute_gae([1.0], [0.0], gamma=0.9, lam=0.0, last_value=0.0)
    tensor([1.0])  # delta_0 = 1 + 0.9*0 - 0 = 1.0; A_0 = 1.0
    """
    # SOLUTION
    T = len(rewards)
    advantages = []
    A = 0.0
    all_values = values + [last_value]  # V(s_0),...,V(s_{T-1}),V(s_T)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * all_values[t + 1] - all_values[t]
        A = delta + gamma * lam * A
        advantages.append(A)

    advantages = list(reversed(advantages))
    return torch.tensor(advantages, dtype=torch.float32)


def test_exercise_3() -> None:
    # Single step, lam=0: A_0 = delta_0 = r + gamma*V_next - V_s
    A = compute_gae([1.0], [0.0], gamma=0.9, lam=0.0, last_value=0.0)
    assert A.shape == (1,), f"Shape must be (1,), got {A.shape}."
    assert torch.isclose(A[0], torch.tensor(1.0)), \
        f"lam=0, single step: A=1.0, got {A[0]}."

    # lam=0: pure TD advantage; verify 2-step
    rewards = [1.0, 1.0]
    values = [0.0, 0.0]
    A_td = compute_gae(rewards, values, gamma=0.9, lam=0.0, last_value=0.0)
    # delta_0 = 1+0.9*0-0=1, delta_1 = 1+0.9*0-0=1
    # A_0 = delta_0 + 0*delta_1 = 1; A_1 = delta_1 = 1
    assert torch.allclose(A_td, torch.tensor([1.0, 1.0])), \
        f"lam=0 advantages must equal TD residuals [1,1], got {A_td}."

    # lam=1, gamma=1: must equal MC advantage = G_t - V(s_t)
    rewards2 = [1.0, 2.0]
    values2 = [0.5, 0.5]
    A_mc = compute_gae(rewards2, values2, gamma=1.0, lam=1.0, last_value=0.0)
    # G_0 = 1+2 = 3; A_0 = 3 - 0.5 = 2.5
    # G_1 = 2;       A_1 = 2 - 0.5 = 1.5
    assert torch.allclose(A_mc, torch.tensor([2.5, 1.5])), \
        f"lam=1 advantages must be MC-based [2.5, 1.5], got {A_mc}."

    # Output dtype
    assert A.dtype == torch.float32, f"dtype must be float32, got {A.dtype}."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: REINFORCE Loss", test_exercise_1),
        ("Exercise 2: Discounted Returns with Baseline", test_exercise_2),
        ("Exercise 3: Generalized Advantage Estimation (GAE)", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

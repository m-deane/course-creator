"""
Module 07 - Advanced Policy Optimization: Self-Check Exercises

Covers:
- PPO clipped surrogate objective
- KL divergence between categorical distributions
- Soft (Polyak) target network parameter update

Run with: python exercises.py
Dependencies: numpy, torch
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Exercise 1: PPO Clipped Objective
# ---------------------------------------------------------------------------

def ppo_clipped_loss(
    log_probs_new: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Compute the PPO clipped surrogate objective.

    Problem
    -------
    PPO constrains policy updates via a clipped importance ratio:

        ratio_t = exp(log_pi_new(a_t|s_t) - log_pi_old(a_t|s_t))

        L_CLIP = mean( min(
            ratio_t * A_t,
            clip(ratio_t, 1-epsilon, 1+epsilon) * A_t
        ))

    We return the *negative* (loss to minimise) for gradient descent.

    Intuition:
      - When A_t > 0 (good action), clipping prevents ratio from
        growing too large (no credit beyond 1+epsilon).
      - When A_t < 0 (bad action), clipping prevents ratio from
        falling below 1-epsilon (no penalty relief beyond that).

    Hints
    -----
    - Compute ratio = (log_probs_new - log_probs_old).exp().
    - torch.clamp for clipping ratio.
    - torch.min for element-wise minimum of the two objectives.

    Parameters
    ----------
    log_probs_new : torch.Tensor, shape (T,)
        Log-probabilities under the new (updated) policy.
    log_probs_old : torch.Tensor, shape (T,)
        Log-probabilities under the old (behaviour) policy.
    advantages : torch.Tensor, shape (T,)
        Advantage estimates A_t (should be normalised externally).
    epsilon : float
        Clipping threshold (e.g., 0.2).

    Returns
    -------
    torch.Tensor
        Scalar loss = -L_CLIP.

    Examples
    --------
    >>> # When new == old, ratio=1; loss = -mean(A_t)
    >>> ppo_clipped_loss(
    ...     torch.tensor([0.0]), torch.tensor([0.0]),
    ...     torch.tensor([2.0]), epsilon=0.2
    ... )
    tensor(-2.0)
    """
    # SOLUTION
    ratio = (log_probs_new - log_probs_old).exp()
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2).mean()
    return loss


def test_exercise_1() -> None:
    # When new == old: ratio=1; clipped=1; loss = -mean(A)
    lp_new = torch.tensor([0.0, 0.0])
    lp_old = torch.tensor([0.0, 0.0])
    adv = torch.tensor([2.0, 4.0])
    loss = ppo_clipped_loss(lp_new, lp_old, adv, epsilon=0.2)
    assert torch.isclose(loss, torch.tensor(-3.0)), \
        f"ratio=1: loss must be -mean(A)=-3.0, got {loss}."

    # Large ratio with positive advantage: clipping activates
    # ratio=exp(1)~2.718, clip to 1.2; loss = -clip*A = -1.2*1 = -1.2
    lp_new2 = torch.tensor([1.0])   # log_new >> log_old
    lp_old2 = torch.tensor([0.0])
    adv2 = torch.tensor([1.0])
    loss2 = ppo_clipped_loss(lp_new2, lp_old2, adv2, epsilon=0.2)
    assert torch.isclose(loss2, torch.tensor(-1.2), atol=1e-5), \
        f"Clipped positive advantage: loss must be -1.2, got {loss2}."

    # Negative advantage: ratio should be clipped at 1-epsilon from above
    # ratio=2.718, but with A<0: surr1=ratio*A (very negative),
    # surr2=clip*A (less negative); min is surr1 (more negative)
    # But clipping prevents ratio from moving to hurt more when A<0:
    # clip to 1.2, surr2=1.2*(-1)=-1.2; surr1=e*(-1)~=-2.718; min=surr1=-2.718
    # loss = -(-2.718) = 2.718 ... wait, that's wrong for PPO's intent.
    # PPO with A<0 and high ratio: min(surr1, surr2) = min(ratio*A, clipped*A)
    # since A<0, ratio*A < clipped*A (more negative); min = ratio*A if ratio>clip
    # Actually PPO clips ratio so it can't RISE when A<0 either:
    # This verifies ratio with negative advantage doesn't exploit
    adv3 = torch.tensor([-1.0])
    loss3 = ppo_clipped_loss(lp_new2, lp_old2, adv3, epsilon=0.2)
    # surr1 = e*(-1); surr2 = 1.2*(-1) = -1.2; min = e*(-1)~-2.72
    # loss = 2.72 approximately
    assert loss3.item() > 0, \
        "Loss must be positive for negative advantages with large ratio update."

    # Loss is scalar
    assert loss.shape == torch.Size([]), f"Loss must be scalar, got {loss.shape}."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: KL Divergence Between Categorical Distributions
# ---------------------------------------------------------------------------

def categorical_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence KL(p || q) for categorical distributions.

    Problem
    -------
    The KL divergence from q to p (measured from p's perspective) is:

        KL(p || q) = sum_a p(a) * log(p(a) / q(a))

    Properties:
      - KL(p || p) = 0 for any p.
      - KL >= 0 always (Gibbs' inequality).
      - KL is NOT symmetric: KL(p||q) != KL(q||p) in general.

    In PPO's early stopping criterion, we monitor KL(pi_old || pi_new)
    and stop updates when it exceeds a threshold (e.g., 0.01).

    Hints
    -----
    - Add a small epsilon (1e-8) inside log() to avoid log(0).
    - Use (p * (p / q + 1e-8).log()).sum() — operates on probability tensors.
    - Input tensors represent probability distributions (already sum to 1).

    Parameters
    ----------
    p : torch.Tensor, shape (A,) or (B, A)
        First distribution (or batch).
    q : torch.Tensor, shape (A,) or (B, A)
        Second distribution (or batch).

    Returns
    -------
    torch.Tensor
        Scalar KL divergence (mean over batch if batched).

    Examples
    --------
    >>> p = torch.tensor([0.9, 0.1])
    >>> q = torch.tensor([0.9, 0.1])
    >>> categorical_kl_divergence(p, q)
    tensor(0.)   # identical distributions
    """
    # SOLUTION
    kl = (p * (p / (q + 1e-8) + 1e-8).log()).sum(dim=-1)
    return kl.mean()


def test_exercise_2() -> None:
    # Identical distributions: KL = 0
    p = torch.tensor([0.7, 0.2, 0.1])
    kl_zero = categorical_kl_divergence(p, p)
    assert torch.isclose(kl_zero, torch.tensor(0.0), atol=1e-6), \
        f"KL(p||p) must be 0, got {kl_zero}."

    # KL >= 0 always
    q = torch.tensor([0.1, 0.6, 0.3])
    kl = categorical_kl_divergence(p, q)
    assert kl.item() >= 0, f"KL divergence must be non-negative, got {kl}."

    # Asymmetry: KL(p||q) != KL(q||p) in general
    kl_pq = categorical_kl_divergence(p, q)
    kl_qp = categorical_kl_divergence(q, p)
    assert not torch.isclose(kl_pq, kl_qp, atol=1e-4), \
        "KL must be asymmetric for different distributions."

    # Scalar output
    assert kl.shape == torch.Size([]), f"KL must be scalar, got {kl.shape}."

    # Batched: mean over batch
    p_batch = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
    q_batch = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
    kl_batch = categorical_kl_divergence(p_batch, q_batch)
    assert torch.isclose(kl_batch, torch.tensor(0.0), atol=1e-6), \
        f"Batch KL of identical distributions must be 0, got {kl_batch}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Soft (Polyak) Target Network Update
# ---------------------------------------------------------------------------

def soft_update(
    target_network: nn.Module,
    online_network: nn.Module,
    tau: float,
) -> None:
    """
    Update target network parameters with Polyak (soft) averaging.

    Problem
    -------
    Hard target updates (copying weights every N steps) cause instability.
    Polyak averaging blends the online and target parameters smoothly:

        theta_target <- tau * theta_online + (1 - tau) * theta_target

    For tau=1.0 this reduces to a hard copy. Typical values: tau=0.005.

    Implement this as an in-place update of target_network's parameters.
    Do NOT update batch-norm running statistics (they are buffers, not
    parameters) — iterate only over named_parameters().

    Hints
    -----
    - Use target_network.named_parameters() and online_network.named_parameters()
      zipped together.
    - target_param.data.mul_(1 - tau).add_(online_param.data * tau) is
      the idiomatic in-place version.
    - Wrap in torch.no_grad() to avoid polluting the autograd graph.

    Parameters
    ----------
    target_network : nn.Module
        Network whose parameters will be updated (in-place).
    online_network : nn.Module
        Source of new parameter values.
    tau : float
        Interpolation factor in [0, 1]. tau=1 means hard copy.

    Returns
    -------
    None
        Modifies target_network parameters in-place.

    Examples
    --------
    >>> # After soft_update with tau=1.0, target == online
    >>> soft_update(target_net, online_net, tau=1.0)
    """
    # SOLUTION
    with torch.no_grad():
        for (_, t_param), (_, o_param) in zip(
            target_network.named_parameters(),
            online_network.named_parameters(),
        ):
            t_param.data.mul_(1.0 - tau).add_(o_param.data * tau)


def test_exercise_3() -> None:
    torch.manual_seed(0)

    def make_linear(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        return nn.Linear(4, 2, bias=False)

    target = make_linear(seed=1)
    online = make_linear(seed=2)

    # tau=1: hard copy -> target must equal online
    soft_update(target, online, tau=1.0)
    for (_, tp), (_, op) in zip(target.named_parameters(), online.named_parameters()):
        assert torch.allclose(tp, op), \
            "tau=1.0 must make target identical to online."

    # tau=0: no change to target
    target2 = make_linear(seed=1)
    original_weights = [p.data.clone() for p in target2.parameters()]
    online2 = make_linear(seed=2)
    soft_update(target2, online2, tau=0.0)
    for p, orig in zip(target2.parameters(), original_weights):
        assert torch.allclose(p.data, orig), \
            "tau=0.0 must not change target parameters."

    # tau=0.5: target must be midpoint of original and online
    target3 = make_linear(seed=1)
    online3 = make_linear(seed=2)
    orig3 = [p.data.clone() for p in target3.parameters()]
    online3_params = [p.data.clone() for p in online3.parameters()]
    soft_update(target3, online3, tau=0.5)
    for p, o, n in zip(target3.parameters(), orig3, online3_params):
        expected = 0.5 * o + 0.5 * n
        assert torch.allclose(p.data, expected, atol=1e-6), \
            f"tau=0.5: target must be midpoint. Expected {expected}, got {p.data}."

    # In-place: target_network object is modified
    target4 = make_linear(seed=1)
    id_before = id(list(target4.parameters())[0])
    online4 = make_linear(seed=2)
    soft_update(target4, online4, tau=0.5)
    id_after = id(list(target4.parameters())[0])
    assert id_before == id_after, "soft_update must modify parameters in-place."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: PPO Clipped Objective", test_exercise_1),
        ("Exercise 2: Categorical KL Divergence", test_exercise_2),
        ("Exercise 3: Soft (Polyak) Target Network Update", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")

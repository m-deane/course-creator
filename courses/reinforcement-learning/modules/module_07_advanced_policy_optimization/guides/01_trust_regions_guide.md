# Trust Region Policy Optimization (TRPO)

## In Brief

Trust region methods constrain how much a policy can change in a single gradient update. Without such a constraint, large updates can collapse performance catastrophically and recovery is slow or impossible. TRPO (Schulman et al., 2015) formalizes this intuition as a constrained optimization problem, providing a principled guarantee that each update improves the policy.

## Key Insight

Gradient descent assumes the loss landscape is locally linear near the current parameters. For policies, this assumption breaks down: a parameter update that looks small in Euclidean space can produce a large behavioral change, because the mapping from parameters to action distributions is nonlinear. Trust region methods measure change in the space of **distributions** rather than parameters, keeping each update within a safe neighborhood where the linear approximation is reliable.

---

## Formal Definition

### The Policy Gradient Problem

A policy $\pi_\theta$ parameterizes a distribution over actions given states. The objective is to maximize the expected discounted return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

Standard gradient ascent updates $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. This works for small step sizes but fails for large ones: the policy may change so drastically that the gradient estimate (computed under $\pi_{\theta_{old}}$) no longer describes the new policy at all.

### KL Divergence as Policy Distance

The Kullback-Leibler divergence measures the distance between two distributions:

$$D_{KL}(\pi_{old} \,\|\, \pi_{new}) = \mathbb{E}_{a \sim \pi_{old}}\left[\log \frac{\pi_{old}(a|s)}{\pi_{new}(a|s)}\right]$$

For policy optimization, the expected KL over the state distribution is used:

$$\overline{D}_{KL}(\pi_{old} \,\|\, \pi_{new}) = \mathbb{E}_{s \sim d^{\pi_{old}}}\left[D_{KL}(\pi_{old}(\cdot|s) \,\|\, \pi_{new}(\cdot|s))\right]$$

where $d^{\pi_{old}}$ is the state visitation distribution under the old policy. This quantity equals zero if and only if the two policies assign identical probabilities to every action in every state.

### TRPO Objective and Constraint (Schulman et al., 2015)

TRPO maximizes a surrogate objective — the importance-weighted advantage — subject to a hard constraint on the KL divergence:

$$\max_\theta \; L(\theta) = \mathbb{E}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a)\right]$$

$$\text{subject to} \quad \mathbb{E}\!\left[D_{KL}\!\left(\pi_{\theta_{old}} \,\|\, \pi_\theta\right)\right] \leq \delta$$

Here $\hat{A}(s,a)$ is the advantage estimate (how much better action $a$ is compared to the average at state $s$), and $\delta$ is a small constant (typically $\delta = 0.01$). The constraint keeps the new policy inside a **trust region** — a ball around the old policy in distribution space.

---

## Intuitive Explanation

Think of policy optimization as navigating a hill in the dark. You can feel which direction is uphill from where you stand, but you cannot see how the hill curves in the distance. Taking a very large step based on local gradient information risks stepping off a cliff.

TRPO says: only take steps small enough that the terrain you measured accurately describes where you are stepping. The KL divergence is your step-length ruler — not in parameter space, but in the space of what the policy actually does. You can reshape the policy distribution's parameters dramatically, but if the resulting distributions are close, you have stayed within the safe neighborhood.

---

## Natural Policy Gradient

The standard gradient $\nabla_\theta J(\theta)$ is measured in Euclidean parameter space, which does not account for the geometry of the distribution manifold. The **Fisher information matrix** $F(\theta)$ defines the local curvature of the distribution manifold:

$$F(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(a|s) \,\nabla_\theta \log \pi_\theta(a|s)^\top\right]$$

The natural policy gradient replaces the Euclidean step with one that respects this geometry:

$$\theta \leftarrow \theta + \alpha \, F(\theta)^{-1} \nabla_\theta J(\theta)$$

This update moves the same distance in distribution space regardless of how the parameters are parameterized. TRPO can be understood as a constrained version of the natural gradient: it solves for the step that maximizes the surrogate objective subject to a KL ball constraint, which yields the natural gradient direction with a step size determined by $\delta$.

---

## Practical Optimization: Conjugate Gradient + Line Search

Computing $F^{-1}$ directly costs $O(d^2)$ in memory and $O(d^3)$ in time for a network with $d$ parameters, making it infeasible for deep networks. TRPO avoids explicit inversion using the **conjugate gradient (CG)** algorithm to solve $Fx = g$ iteratively, requiring only matrix-vector products $Fv$ rather than the full matrix.

After computing the update direction $x = F^{-1}g$, a **backtracking line search** finds the largest step size that satisfies the KL constraint and achieves a positive surrogate objective improvement.

```python
def trpo_step(policy, states, actions, advantages, old_log_probs,
              delta=0.01, cg_iters=10, backtrack_coeff=0.8,
              backtrack_iters=10):
    """
    One TRPO policy update.

    Args:
        policy:        Neural network policy (callable)
        states:        Tensor of shape (N, obs_dim)
        actions:       Tensor of shape (N, act_dim)
        advantages:    Tensor of shape (N,) — normalized advantage estimates
        old_log_probs: Tensor of shape (N,) — log pi_old(a|s)
        delta:         KL divergence constraint radius
        cg_iters:      Conjugate gradient iterations
        backtrack_coeff: Line search shrinkage factor
        backtrack_iters: Maximum line search steps

    Returns:
        Updated policy parameters (in-place) and diagnostics dict
    """
    import torch

    # --- 1. Compute surrogate objective gradient ---
    new_log_probs = policy.log_prob(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surrogate = (ratio * advantages).mean()
    g = torch.autograd.grad(surrogate, policy.parameters())
    g = torch.cat([p.view(-1) for p in g])

    # --- 2. Conjugate gradient: solve F*x = g ---
    def fisher_vector_product(v):
        """Compute F*v without forming F explicitly."""
        kl = policy.kl_divergence(states)           # KL(pi_old || pi_new)
        kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        kl_grad = torch.cat([p.view(-1) for p in kl_grad])
        jvp = (kl_grad * v).sum()
        fvp = torch.autograd.grad(jvp, policy.parameters())
        return torch.cat([p.view(-1) for p in fvp]).detach()

    step_direction = conjugate_gradient(fisher_vector_product, g, cg_iters)

    # --- 3. Compute natural step size from KL constraint ---
    sFs = (step_direction * fisher_vector_product(step_direction)).sum()
    step_size = torch.sqrt(2 * delta / (sFs + 1e-8))
    full_step = step_size * step_direction

    # --- 4. Backtracking line search ---
    old_params = torch.cat([p.data.view(-1) for p in policy.parameters()])
    for i in range(backtrack_iters):
        candidate = old_params + (backtrack_coeff ** i) * full_step
        _set_params(policy, candidate)

        new_log_probs = policy.log_prob(states, actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        new_surrogate = (ratio * advantages).mean()
        kl = policy.kl_divergence(states).mean()

        if new_surrogate > surrogate and kl <= delta:
            return {"success": True, "kl": kl.item(), "steps": i + 1}

    # Revert on failure
    _set_params(policy, old_params)
    return {"success": False, "kl": None, "steps": backtrack_iters}


def conjugate_gradient(Av, b, n_iters=10, tol=1e-10):
    """
    Solve Ax = b approximately using the conjugate gradient method.
    Only requires matrix-vector products Av, not the matrix A itself.
    """
    import torch
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot = (r * r).sum()

    for _ in range(n_iters):
        Ap = Av(p)
        alpha = r_dot / ((p * Ap).sum() + 1e-8)
        x = x + alpha * p
        r = r - alpha * Ap
        new_r_dot = (r * r).sum()
        if new_r_dot < tol:
            break
        beta = new_r_dot / (r_dot + 1e-8)
        p = r + beta * p
        r_dot = new_r_dot

    return x
```

---

## Diagram: Trust Region vs Unconstrained Gradient

```
Parameter space view:            Distribution space view:

         θ_new (bad)                    π_new (bad)
            *                            far from π_old
           /
          / large                  [trust region boundary]
         /  step                       ___________
        *----------*           π_old |           |
     θ_old       θ_new         *     |  π_new    |
     (good)      (OK)                | (good)    |
                                     |___________|

Unconstrained step moves far      KL constraint keeps π_new
in parameter space, may            close in distribution space
destroy policy behavior            regardless of parameter norm
```

---

## Why TRPO Works but Is Complex to Implement

TRPO provides a **monotonic improvement guarantee** under certain assumptions: each constrained update cannot decrease the true objective $J(\theta)$. This makes it theoretically sound, unlike plain policy gradient which can oscillate or diverge.

The complexity cost is substantial:

| Requirement | Cost |
|---|---|
| Fisher-vector products | Two backward passes per CG iteration |
| CG solver | 10-50 iterations per update |
| Backtracking line search | 10-20 forward passes per update |
| Constraint verification | KL computation every step |
| Memory overhead | Full gradient stored throughout CG |

For large networks this translates to 10-50x the compute of a plain gradient step. In practice, TRPO is rarely the first choice today — PPO (Guide 02) achieves similar performance at a fraction of the implementation complexity.

---

## Common Pitfalls

**Pitfall 1 — KL constraint too loose.**
Setting $\delta$ too large (e.g., $\delta = 0.5$) defeats the purpose: the trust region becomes so wide that catastrophic updates are not prevented. Typical values are $\delta \in [0.01, 0.05]$. If your policy collapses early in training, decrease $\delta$.

**Pitfall 2 — Advantage estimates not normalized.**
TRPO's surrogate is sensitive to the scale of $\hat{A}$. If advantages are not normalized to zero mean and unit variance per batch, the effective step size varies wildly across batches. Always normalize advantages before computing the ratio objective.

**Pitfall 3 — CG solver precision.**
Insufficient CG iterations produce a poor approximation of $F^{-1}g$, causing the step direction to violate the KL constraint. Use 10 CG iterations as a minimum; monitor the residual norm to verify convergence.

**Pitfall 4 — Fisher-vector product with wrong computation graph.**
The KL divergence used in the FVP must be computed under the current parameters, not detached. Using `.detach()` too early removes the computation graph needed for the second-order gradient. This is the most common implementation bug in TRPO.

**Pitfall 5 — Confusing the constraint direction.**
The constraint is $D_{KL}(\pi_{\theta_{old}} \| \pi_\theta)$, i.e., the old policy as the reference distribution. Some implementations accidentally reverse the arguments. This matters because KL divergence is asymmetric: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$.

---

## Connections

- **Builds on:** Policy gradient theorem (Module 5), advantage estimation (Module 6), importance sampling
- **Leads to:** PPO (Guide 02) which approximates TRPO's constraint with clipping; natural gradients in variational inference
- **Related to:** Proximal optimization methods, mirror descent, information geometry

---

## Further Reading

- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2015). *Trust Region Policy Optimization.* ICML 2015. — The primary reference for this guide; Section 3 contains the monotonic improvement theorem.
- Kakade, S. M. (2002). *A Natural Policy Gradient.* NeurIPS 2001. — Original derivation of the natural policy gradient.
- Martens, J. (2014). *New Insights and Perspectives on the Natural Gradient Method.* JMLR. — Comprehensive treatment of Fisher information and natural gradients.
- Schulman, J. (2016). *Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs.* PhD Thesis, UC Berkeley. — Extended derivations and connections to PPO.

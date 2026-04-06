# Proximal Policy Optimization (PPO)

> **Reading time:** ~14 min | **Module:** 7 — Advanced Policy Optimization | **Prerequisites:** Module 6

## In Brief

Proximal Policy Optimization (Schulman et al., 2017) achieves the stability of TRPO without the computational overhead of conjugate gradient and line search. Instead of enforcing a hard KL constraint, PPO clips the importance ratio to prevent large policy updates. The result is a first-order algorithm that is easy to implement, fast to run, and competitive with TRPO on nearly every benchmark.

<div class="callout-key">

<strong>Key Concept:</strong> Proximal Policy Optimization (Schulman et al., 2017) achieves the stability of TRPO without the computational overhead of conjugate gradient and line search. Instead of enforcing a hard KL constraint, PPO clips the importance ratio to prevent large policy updates.

</div>


## Key Insight

TRPO's KL constraint stops updates that move the policy too far. PPO approximates the same behavior by simply clipping the importance ratio $r_t(\theta)$ into the interval $[1-\epsilon, 1+\epsilon]$. Once the ratio hits the clip boundary, the gradient through that transition is zeroed — so continuing to push the policy further in that direction produces no additional objective gain. The policy learns to stay inside the trust region without explicitly enforcing a constraint.

---


<div class="callout-key">

<strong>Key Point:</strong> TRPO's KL constraint stops updates that move the policy too far.

</div>
## Intuitive Explanation

Imagine a hiring manager reviewing candidate scores. If a candidate scores 20% above the threshold, you hire them. If they score 100% above the threshold, you still just hire them — the extra margin gives you no additional benefit. PPO applies the same logic to policy updates: once you have improved an action's probability by $\epsilon$, further increase yields no reward gradient. The policy has no incentive to make radical changes in a single update.

---


<div class="callout-key">

<strong>Key Point:</strong> Imagine a hiring manager reviewing candidate scores.

</div>
## Formal Definition

### The Probability Ratio

<div class="callout-info">

<strong>Info:</strong> ### The Probability Ratio

PPO builds on the same importance-weighted surrogate as TRPO.

</div>


PPO builds on the same importance-weighted surrogate as TRPO. Define the probability ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

When $r_t = 1$, the new policy assigns the same probability as the old policy to action $a_t$ in state $s_t$. When $r_t > 1$, the new policy finds this action more likely; when $r_t < 1$, less likely.

### PPO-Clip Objective (Schulman et al., 2017)

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\right)\right]$$

The $\min$ of two terms is the crucial mechanism:

- **Term 1:** $r_t(\theta)\hat{A}_t$ — the unclipped surrogate (TRPO's objective)
- **Term 2:** $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t$ — the clipped surrogate

Taking the minimum of the two means the objective is **pessimistic**: it takes whichever term is lower, so any update that would benefit from going beyond the clip boundary is not rewarded.

### How Clipping Works by Case

**Case 1: Positive advantage ($\hat{A}_t > 0$), action is being increased ($r_t > 1+\epsilon$):**

$$\min(r_t \hat{A}_t,\; (1+\epsilon)\hat{A}_t) = (1+\epsilon)\hat{A}_t$$

The gradient is zeroed past the clip boundary — no incentive to increase the ratio further.

**Case 2: Negative advantage ($\hat{A}_t < 0$), action is being decreased ($r_t < 1-\epsilon$):**

$$\min(r_t \hat{A}_t,\; (1-\epsilon)\hat{A}_t) = (1-\epsilon)\hat{A}_t$$

Again, no gradient benefit from decreasing the ratio past the lower clip boundary.

**Case 3: Ratio within $[1-\epsilon, 1+\epsilon]$:**

Both terms are equal; the clipped and unclipped objectives agree. Normal gradient flow.

---


## Clip Behavior Diagram

```
                        L^CLIP(θ)
                            |
  Positive advantage (Â > 0):
                            |      ______ slope = Â (clipped, no gradient)
               slope = Â   |     /
                           /     /
         ________________/     /
        |                |    /
  ------+----------------+---/---------> r_t(θ)
        0             1-ε  1  1+ε
                            slope = Â here (unclipped)

  Negative advantage (Â < 0):
                            |
       slope = Â (clipped)  |
   \___                     |
        \                   |
         \                  |
  --------\-----------------+----------> r_t(θ)
           1-ε              1  1+ε
                     slope = Â (unclipped)

Key: The objective is FLAT beyond the clip boundaries in the direction
     that would make the policy change larger. Gradient is zero there.
```

---

## Full PPO Objective

In practice, PPO trains a combined actor-critic and adds an entropy bonus:

$$L^{PPO}(\theta) = \mathbb{E}_t\!\left[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 \mathcal{H}[\pi_\theta](\cdot|s_t)\right]$$

Where:
- $L^{VF}(\theta) = \left(V_\theta(s_t) - V_t^{targ}\right)^2$ is the value function loss
- $\mathcal{H}[\pi_\theta](\cdot|s_t)$ is the policy entropy (encourages exploration)
- $c_1 \approx 0.5$ is the value loss coefficient
- $c_2 \approx 0.01$ is the entropy bonus coefficient

---

## PPO Practical Implementation

### Data Collection and Multiple Epochs

PPO collects a batch of experience under $\pi_{\theta_{old}}$, then performs **multiple gradient epochs** over the same batch. This amortizes the environment interaction cost:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical, Normal


class PPOActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    Shares a feature extractor trunk between policy and value heads.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Policy head: outputs logits for categorical actions
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        # Value head: outputs scalar state value
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.trunk(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, obs):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


def ppo_clip_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    """
    Compute the PPO-Clip objective.

    Args:
        log_probs_new: log π_θ(a|s) under current policy — shape (N,)
        log_probs_old: log π_θ_old(a|s) when data was collected — shape (N,)
        advantages:    normalized advantage estimates — shape (N,)
        epsilon:       clip ratio (typically 0.2)

    Returns:
        Scalar loss (negated for gradient ascent via minimization)
    """
    # Importance ratio: how much more/less likely under new vs old policy
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Unclipped surrogate: standard importance-weighted advantage
    surr1 = ratio * advantages

    # Clipped surrogate: ratio is constrained to [1-eps, 1+eps]
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages

    # Pessimistic (lower) bound: min of the two
    clip_loss = -torch.min(surr1, surr2).mean()   # negative for minimization
    return clip_loss


def ppo_update(model, optimizer, rollout, n_epochs=4, batch_size=64,
               epsilon=0.2, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """
    Run PPO update over collected rollout data.

    Args:
        model:         PPOActorCritic instance
        optimizer:     Adam optimizer bound to model
        rollout:       dict with keys: states, actions, log_probs_old,
                       returns, advantages — all shape (T,)
        n_epochs:      Number of gradient epochs over same rollout
        batch_size:    Mini-batch size for stochastic gradient
        epsilon:       PPO clip parameter
        vf_coef:       Value function loss coefficient c1
        ent_coef:      Entropy bonus coefficient c2
        max_grad_norm: Gradient clipping norm

    Returns:
        dict of training diagnostics
    """
    states       = torch.FloatTensor(rollout["states"])
    actions      = torch.LongTensor(rollout["actions"])
    log_probs_old = torch.FloatTensor(rollout["log_probs_old"])
    returns      = torch.FloatTensor(rollout["returns"])
    advantages   = torch.FloatTensor(rollout["advantages"])

    # Normalize advantages: critical for stable PPO training
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    N = len(states)
    diagnostics = {"policy_loss": [], "value_loss": [], "entropy": [],
                   "clip_fraction": [], "approx_kl": []}

    for epoch in range(n_epochs):
        # Random permutation for mini-batch stochastic gradient
        indices = torch.randperm(N)

        for start in range(0, N, batch_size):
            idx = indices[start:start + batch_size]
            mb_states    = states[idx]
            mb_actions   = actions[idx]
            mb_log_old   = log_probs_old[idx]
            mb_returns   = returns[idx]
            mb_advantages = advantages[idx]

            # Evaluate current policy on mini-batch
            log_probs_new, values, entropy = model.evaluate(mb_states, mb_actions)

            # PPO-Clip policy loss
            policy_loss = ppo_clip_loss(log_probs_new, mb_log_old,
                                        mb_advantages, epsilon)

            # Value function MSE loss
            value_loss = 0.5 * ((values - mb_returns) ** 2).mean()

            # Combined loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping prevents very large individual updates
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Diagnostics: clip fraction and approximate KL divergence
            with torch.no_grad():
                ratio = torch.exp(log_probs_new - mb_log_old)
                clip_frac = ((ratio - 1.0).abs() > epsilon).float().mean()
                approx_kl = ((log_probs_old[idx] - log_probs_new).mean())

            diagnostics["policy_loss"].append(policy_loss.item())
            diagnostics["value_loss"].append(value_loss.item())
            diagnostics["entropy"].append(entropy.mean().item())
            diagnostics["clip_fraction"].append(clip_frac.item())
            diagnostics["approx_kl"].append(approx_kl.item())

    return {k: np.mean(v) for k, v in diagnostics.items()}
```

</div>

### Advantage Estimation (GAE)

PPO uses Generalized Advantage Estimation (Schulman et al., 2016) to compute $\hat{A}_t$:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation.

    Combines TD(0) (low variance, high bias) with MC returns (low bias,
    high variance) via exponential weighting controlled by lam.
    lam=0 → TD(0) residuals; lam=1 → full Monte Carlo advantage.

    Args:
        rewards: array of shape (T,)
        values:  array of shape (T+1,)  — V(s_t) for t=0..T
        dones:   array of shape (T,)    — 1 if episode ended at t
        gamma:   discount factor
        lam:     GAE lambda

    Returns:
        advantages: array of shape (T,)
        returns:    array of shape (T,)   — advantages + values (TD targets)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] * (1 - dones[t])     # zero at episode end
        delta = rewards[t] + gamma * next_value - values[t]   # TD residual
        gae = delta + gamma * lam * (1 - dones[t]) * gae      # GAE recursion
        advantages[t] = gae

    returns = advantages + values[:T]    # TD targets for value function
    return advantages, returns
```

</div>

---

## PPO vs TRPO Comparison

| Dimension | TRPO | PPO |
|---|---|---|
| Constraint enforcement | Hard KL constraint | Soft (clipping) |
| Optimization method | Conjugate gradient + line search | Adam (first-order) |
| Backward passes per step | ~20 | 1 |
| Implementation complexity | High | Low |
| Memory overhead | High (CG vectors) | Low |
| Theoretical guarantee | Monotonic improvement bound | Empirical stability |
| Performance (MuJoCo) | Competitive | Competitive (slight edge in some envs) |
| Industry adoption | Research baseline | Default choice |
| Code lines (minimal) | ~300 | ~100 |

**When to choose TRPO over PPO:** When you need a provable safety guarantee (e.g., safety-critical robotics research) or when you are ablating the effect of the exact vs. approximate constraint. For all practical purposes, PPO is preferred.

---


<div class="compare">
<div class="compare-card">
<div class="header before">PPO</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">TRPO Comparison</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Hyperparameter Guide

### Clip Parameter $\epsilon$

Controls the width of the trust region. Smaller $\epsilon$ → more conservative updates.

| $\epsilon$ | Effect | When to use |
|---|---|---|
| $0.1$ | Conservative; slower learning but very stable | Sensitive environments, fine-tuning |
| $0.2$ | Default; good balance | Most environments |
| $0.3$ | Aggressive; faster learning but less stable | Early exploration, simple envs |

### Number of Epochs

More epochs amortize data collection but increase the chance of overfitting the old batch.

| Epochs | Effect |
|---|---|
| $3$ | Conservative; useful when data collection is cheap |
| $4{-}10$ | Typical range; monitor clip fraction and approx KL |
| $> 10$ | Risk of policy drift; verify approx KL stays below 0.02 |

### Other Critical Hyperparameters

| Parameter | Typical Value | Notes |
|---|---|---|
| Mini-batch size | $64{-}2048$ | Larger batches → more stable gradient estimates |
| Learning rate | $3 \times 10^{-4}$ | Adam; may decay linearly to 0 |
| Value loss coefficient $c_1$ | $0.5$ | Reduce if value loss dominates |
| Entropy coefficient $c_2$ | $0.01$ | Increase if policy collapses to deterministic |
| Gradient clip norm | $0.5$ | Essential; prevents rare but catastrophic spikes |
| GAE $\lambda$ | $0.95$ | Bias-variance tradeoff in advantage estimation |
| Discount $\gamma$ | $0.99$ | Standard; reduce for shorter-horizon problems |

### Diagnostic Signals

Monitor these quantities during training to detect problems early:

| Diagnostic | Healthy Range | Problem if... |
|---|---|---|
| Approx KL divergence | $< 0.02$ | $> 0.05$ → learning rate too high or epsilon too large |
| Clip fraction | $0.1{-}0.3$ | $> 0.5$ → epsilon too small or LR too high |
| Policy entropy | Decreasing slowly | Drops to near 0 → increase $c_2$ |
| Value loss | Decreasing | Increasing → value function not learning |
| Explained variance | $> 0.5$ | $< 0$ → value function worse than mean predictor |

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1 — Advantages not normalized per mini-batch.**
Normalizing advantages once over the full rollout is incorrect when using mini-batches. The statistics should be computed over the rollout (not per mini-batch), but mini-batches should use the rollout-level mean and std. Normalizing per mini-batch reintroduces scale noise.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1 — Advantages not normalized per mini-batch.**
Normalizing advantages once over the full rollout is incorrect when using mini-batches.

</div>

**Pitfall 2 — Recomputing old log-probabilities.**
The $\pi_{\theta_{old}}(a_t|s_t)$ values must be stored at collection time and remain fixed throughout all epochs. If you call the current model to get "old" log probs, you are always computing ratio = 1, and PPO degenerates to vanilla policy gradient.

**Pitfall 3 — Forgetting gradient clipping.**
Even with PPO clipping, individual gradient steps can have very large norms due to mini-batch noise. Always apply `clip_grad_norm_` with a norm of 0.5 before the optimizer step.

**Pitfall 4 — Too many epochs with large $\epsilon$.**
With $\epsilon = 0.3$ and 20 epochs, the policy drifts far from the old policy that generated the data. The importance ratio no longer corrects for this, causing off-policy errors. Keep epochs * clip_fraction product small.

**Pitfall 5 — Value function target not detached.**
The returns (TD targets) used in the value loss must not be part of the computation graph. Compute them from the old value network and detach before passing to the loss. Otherwise the value loss gradient flows back through the returns, destabilizing training.

**Pitfall 6 — Ignoring episode termination in GAE.**
When a trajectory is truncated (time limit hit, not episode end), the last value should bootstrap: $V_T$ from the value network, not 0. Using 0 at truncation introduces large bias in long-horizon environments. Check the `terminated` vs `truncated` flags from Gymnasium.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** TRPO (Guide 01) — PPO is TRPO with the hard constraint replaced by a soft penalty; policy gradient theorem (Module 5); GAE advantage estimation (Module 6)
- **Leads to:** SAC (Guide 03) for continuous control; PPO is the base algorithm for RLHF fine-tuning of large language models
- **Related to:** Proximal point methods in convex optimization; trust region methods in numerical optimization; clipped surrogate objectives in off-policy learning

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347. — The primary reference; read Section 3 for the full objective and Section 5 for hyperparameter discussion.
- Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016. — GAE derivation; essential companion for understanding $\hat{A}_t$.
- Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). *Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO.* ICLR 2020. — Shows that implementation details (normalization, clipping, etc.) matter more than the objective itself.
- OpenAI Spinning Up — PPO implementation: https://spinningup.openai.com/en/latest/algorithms/ppo.html — Clean reference implementation with extensive documentation.


---

## Cross-References

<a class="link-card" href="./02_ppo_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_ppo_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>

# REINFORCE: Monte Carlo Policy Gradient

> **Reading time:** ~10 min | **Module:** 6 — Policy Gradient | **Prerequisites:** Module 5

## In Brief

REINFORCE (Williams, 1992) is the simplest practical instantiation of the policy gradient theorem. It estimates the action-value $Q^{\pi_\theta}(s,a)$ using the complete Monte Carlo return $G_t$ from each time step in an episode, then uses the score-weighted return to update policy parameters via gradient ascent.

<div class="callout-key">

<strong>Key Concept:</strong> REINFORCE (Williams, 1992) is the simplest practical instantiation of the policy gradient theorem. It estimates the action-value $Q^{\pi_\theta}(s,a)$ using the complete Monte Carlo return $G_t$ from each time step in an episode, then uses the score-weighted return to update policy parameters via gradient ascent.

</div>


## Key Insight

Replace the unknown $Q^{\pi_\theta}(s,a)$ in the policy gradient theorem with its unbiased Monte Carlo estimate: the actual discounted return $G_t$ observed by running the policy to completion. No value function is required — only sampled trajectories.

---


<div class="callout-key">

<strong>Key Point:</strong> Replace the unknown $Q^{\pi_\theta}(s,a)$ in the policy gradient theorem with its unbiased Monte Carlo estimate: the actual discounted return $G_t$ observed by running the policy to completion.

</div>

## Intuitive Explanation

Think of REINFORCE as a trial-and-error learner that watches entire episodes play out before updating:

<div class="callout-key">

<strong>Key Point:</strong> Think of REINFORCE as a trial-and-error learner that watches entire episodes play out before updating:

1.

</div>


1. Run a complete episode following the current policy.
2. For each step, compute how much total reward followed from that point forward ($G_t$).
3. If $G_t$ was large, increase the log-probability of the action taken — that action sequence worked.
4. If $G_t$ was small (or negative), decrease the log-probability — that action sequence did not work.
5. Repeat many episodes, averaging out the stochasticity.

The intuition parallels human trial-and-error learning: you complete a task, recall which decisions led to good outcomes, and make those decisions more likely next time.

---


## Formal Definition

Given a trajectory $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T)$ sampled under $\pi_\theta$, define the **return from time $t$**:

<div class="callout-info">

<strong>Info:</strong> Given a trajectory $\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T)$ sampled under $\pi_\theta$, define the **return from time $t$**:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1...

</div>


$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

The **REINFORCE update rule** is:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

where the $\gamma^t$ factor discounts the gradient contribution of later time steps. In the episodic, undiscounted case ($\gamma = 1$), this simplifies to weighting each score function by the episode return from that step.

---


## Derivation from the Policy Gradient Theorem

Starting from the policy gradient theorem (Guide 01):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot Q^{\pi_\theta}(S_t,A_t)\right]$$

Since $G_t$ is an unbiased estimate of $Q^{\pi_\theta}(S_t,A_t)$ (by definition of the action-value function under $\pi$):

$$\mathbb{E}_{\pi_\theta}[G_t \mid S_t, A_t] = Q^{\pi_\theta}(S_t, A_t)$$

Substituting and estimating the expectation with a single trajectory:

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

This is an unbiased (but high-variance) gradient estimator.

---

## Algorithm Pseudocode

```
REINFORCE Algorithm (Williams, 1992)
────────────────────────────────────
Input: differentiable policy π(a|s; θ), step size α, discount γ
Initialize: θ arbitrarily

Repeat for each episode:
    1. Generate trajectory τ = (S₀,A₀,R₁,S₁,A₁,R₂,...,S_{T-1},A_{T-1},R_T)
       by following π(·|·; θ)

    2. For t = 0, 1, ..., T-1:
           Compute G_t = Σ_{k=0}^{T-t-1} γ^k · R_{t+k+1}

    3. For t = 0, 1, ..., T-1:
           θ ← θ + α · γ^t · G_t · ∇_θ log π(A_t|S_t; θ)

Until θ converges (or training budget exhausted)
```

Note: steps 2 and 3 are combined in practice (compute $G_t$ backward from $T$, then update forward from $0$).

---

## REINFORCE with Baseline

### The Variance Problem

Although the REINFORCE gradient estimator is unbiased, its variance is high:

$$\text{Var}[G_t \nabla \log \pi_\theta] \approx \mathbb{E}[G_t^2 |\nabla \log \pi_\theta|^2] - (\nabla J)^2$$

For long episodes with large rewards, $G_t$ can vary by orders of magnitude across trajectories. This causes erratic gradient steps and slow learning.

### Baseline Subtraction

We can subtract any function $b(S_t)$ that does not depend on the action $A_t$ without introducing bias:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_t \gamma^t (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)\right]$$

**Proof of unbiasedness:**

$$\mathbb{E}_{A_t \sim \pi_\theta}\!\left[b(S_t) \nabla_\theta \log \pi_\theta(A_t|S_t)\right] = b(S_t) \nabla_\theta \underbrace{\sum_a \pi_\theta(a|S_t)}_{=1} = 0$$

The baseline has zero expected contribution because it is constant with respect to the action.

### REINFORCE with Baseline Update Rule

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)$$

### Optimal Baseline

The variance-minimizing baseline is:

$$b^*(s) = \frac{\mathbb{E}[G_t^2 |\nabla \log \pi_\theta|^2]}{\mathbb{E}[|\nabla \log \pi_\theta|^2]}$$

In practice, this is approximated as a constant average return $\bar{G}$ or learned as a state-value function $V(s)$.

---

## The Advantage Function

Setting $b(s) = V^{\pi_\theta}(s)$ gives the **advantage function**:

$$A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$$

which measures how much better (or worse) action $a$ is compared to the average action in state $s$ under the current policy.

The REINFORCE with baseline update becomes:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t (G_t - V(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)$$

where $V(S_t)$ is estimated by a learned baseline network (trained on the same trajectories) or a running average.

**Interpretation:** Only update strongly when the return is surprisingly good or surprisingly bad. When the action performed exactly as expected (return $\approx V(s)$), make no change.

---

## Code Snippet


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    """Simple softmax policy for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(obs), dim=-1)

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        probs = self.forward(obs_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted returns G_t = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}.

    Computed backward from the end of the episode for efficiency.
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G       # G_t = R_{t+1} + γ * G_{t+1}
        returns.insert(0, G)    # prepend to maintain temporal order
    return returns


def reinforce_episode(
    env: gym.Env,
    policy: PolicyNetwork,
    optimizer: optim.Optimizer,
    gamma: float = 0.99,
    baseline: float = 0.0,      # constant baseline; set to running mean in practice
) -> float:
    """
    Run one REINFORCE episode and update policy parameters.

    Returns the total undiscounted episode return.
    """
    log_probs = []
    rewards = []

    # --- Step 1: Generate trajectory ---
    obs, _ = env.reset()
    done = False
    while not done:
        action, log_prob = policy.select_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        done = terminated or truncated

    # --- Step 2: Compute returns ---
    returns = compute_returns(rewards, gamma)
    returns_t = torch.FloatTensor(returns)

    # --- Optional: normalize returns (additional variance reduction) ---
    if returns_t.std() > 1e-8:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    # --- Step 3: Compute REINFORCE loss and update ---
    # Loss = -Σ_t γ^t (G_t - b) log π(A_t|S_t)
    # Negated for gradient descent (we want gradient ascent on J)
    T = len(rewards)
    discounts = torch.FloatTensor([gamma**t for t in range(T)])
    loss = -torch.stack([
        discounts[t] * (returns_t[t] - baseline) * log_probs[t]
        for t in range(T)
    ]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return sum(rewards)


def train_reinforce(
    env_name: str = "CartPole-v1",
    n_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-3,
) -> List[float]:
    """Train a policy using REINFORCE with a running average baseline."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    baseline = 0.0       # running mean of returns
    baseline_alpha = 0.05  # exponential moving average coefficient

    for ep in range(n_episodes):
        total_return = reinforce_episode(env, policy, optimizer, gamma, baseline)
        episode_returns.append(total_return)

        # Update baseline using exponential moving average
        baseline = (1 - baseline_alpha) * baseline + baseline_alpha * total_return

        if (ep + 1) % 100 == 0:
            mean_return = np.mean(episode_returns[-100:])
            print(f"Episode {ep+1}: mean return (last 100) = {mean_return:.1f}")

    env.close()
    return episode_returns
```

</div>

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1 — High variance without baseline.**
Vanilla REINFORCE (no baseline) requires hundreds to thousands of episodes to converge even on simple environments like CartPole. Always include at least a constant baseline (running mean of returns) and ideally a learned $V(s)$ baseline.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1 — High variance without baseline.**
Vanilla REINFORCE (no baseline) requires hundreds to thousands of episodes to converge even on simple environments like CartPole.

</div>

**Pitfall 2 — Using future rewards only (causality).**
A subtle but important optimization: action $A_t$ cannot have caused rewards before time $t$. The theoretically correct update uses only the causal return $G_t = \sum_{k \geq t}$, not the full episode return $G_0$. Some implementations mistakenly use $G_0$ for all steps, which is also unbiased (the extra terms cancel in expectation) but has higher variance.

**Pitfall 3 — Forgetting the $\gamma^t$ discount factor.**
The update rule is $\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla \log \pi$, not $\alpha G_t \nabla \log \pi$. The $\gamma^t$ factor appears because the policy gradient theorem is stated for the discounted-start formulation. In practice, many implementations drop $\gamma^t$ (the undiscounted approximation) and still work well, but the omission should be explicit.

**Pitfall 4 — Needs complete episodes.**
REINFORCE cannot update until the episode ends because $G_t$ requires all future rewards. This makes it inapplicable to continuing (non-episodic) tasks and slow when episodes are long. Actor-critic methods (Guide 03) fix this by bootstrapping with a value estimate.

**Pitfall 5 — Return normalization can obscure the baseline.**
Normalizing returns $(G_t - \text{mean}(G)) / \text{std}(G)$ within a batch is a useful variance reduction trick, but it effectively changes the baseline on every update and should not be combined naively with a separately learned baseline $V(s)$. Use one or the other, not both.

**Pitfall 6 — Episode length variations bias learning.**
If some episodes are long (many steps) and others short, the gradient estimator is not normalized consistently. Use per-step averaging rather than per-episode sum when mixing different episode lengths.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Policy gradient theorem (Guide 01), Monte Carlo methods (Module 02)
- **Leads to:** Actor-critic methods (Guide 03), which replace $G_t$ with a bootstrapped value estimate
- **Related to:** REINFORCE as a score function estimator appears in variational inference (ELBO gradient) and as the REINFORCE trick in latent variable models

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256 — the original REINFORCE paper; short, readable, foundational
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 13.3 — REINFORCE with baseline, variance analysis, convergence properties
- Greensmith, E., Bartlett, P. L., & Baxter, J. (2004). Variance reduction techniques for gradient estimates in reinforcement learning. *JMLR* — rigorous treatment of baselines and variance bounds
- Peters, J. & Schaal, S. (2008). Reinforcement learning of motor skills with policy gradients. *Neural Networks* — practical guide to policy gradient for robotics with Gaussian policies


---

## Cross-References

<a class="link-card" href="./02_reinforce_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_reinforce_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>

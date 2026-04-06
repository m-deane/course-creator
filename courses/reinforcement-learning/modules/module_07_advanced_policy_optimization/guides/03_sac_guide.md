# Soft Actor-Critic (SAC)

> **Reading time:** ~13 min | **Module:** 7 — Advanced Policy Optimization | **Prerequisites:** Module 6

## In Brief

Soft Actor-Critic (Haarnoja et al., 2018) is an off-policy actor-critic algorithm built on the **maximum entropy reinforcement learning** framework. Rather than simply maximizing expected return, SAC maximizes a combination of return and policy entropy. This makes the learned policy stochastically exploratory by design, producing policies that are more robust, better at avoiding local optima, and more sample-efficient than on-policy methods like PPO.

<div class="callout-key">
<strong>Key Concept:</strong> Soft Actor-Critic (Haarnoja et al., 2018) is an off-policy actor-critic algorithm built on the **maximum entropy reinforcement learning** framework. Rather than simply maximizing expected return, SAC maximizes a combination of return and policy entropy.
</div>


## Key Insight

Standard RL trains a policy to be as deterministic as possible — commit to the best action and ignore all others. Maximum entropy RL says: be as random as you can while still collecting good rewards. This seems counterintuitive, but it has a profound effect: a policy with high entropy explores broadly, avoids brittle commitments to single strategies, and transfers more easily to perturbed environments.

---



<div class="callout-key">
<strong>Key Point:</strong> Standard RL trains a policy to be as deterministic as possible — commit to the best action and ignore all others.
</div>
## Intuitive Explanation

Imagine training a robot arm to grasp objects. A standard RL policy commits to one specific joint trajectory. If the object is slightly displaced, the policy fails because it has no experience with variations. A maximum entropy policy explores many different grasping approaches simultaneously — it finds multiple ways to succeed. When the object moves, it has backup strategies available.

<div class="callout-key">
<strong>Key Point:</strong> Imagine training a robot arm to grasp objects.
</div>


The temperature $\alpha$ adjusts how much the robot "cares" about trying different approaches versus optimizing the reward. High $\alpha$ → diverse strategies, slower convergence to optimal. Low $\alpha$ → focused behavior, closer to standard RL.

---


## Formal Definition

### Maximum Entropy Reinforcement Learning

<div class="callout-info">
<strong>Info:</strong> ### Maximum Entropy Reinforcement Learning

The standard RL objective is:

$$J_{standard}(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[r(s_t, a_t)\right]$$

The maximum entropy RL obje...
</div>


The standard RL objective is:

$$J_{standard}(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[r(s_t, a_t)\right]$$

The maximum entropy RL objective augments this with a policy entropy term at every step:

$$J(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

where $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}\!\left[\log \pi(a|s_t)\right]$ is the Shannon entropy of the policy distribution at state $s_t$, and $\alpha > 0$ is the **temperature parameter** controlling the relative importance of entropy versus reward. (Note: $\alpha$ here denotes the SAC temperature, not the learning rate $\alpha$ used in earlier modules.)

### The Soft Bellman Equations

In the maximum entropy framework, the soft Q-function satisfies:

$$Q^{\pi}(s_t, a_t) = \mathbb{E}\!\left[\sum_{l=0}^{\infty} \gamma^l \left(r_{t+l} + \alpha \mathcal{H}(\pi(\cdot|s_{t+l+1}))\right)\right]$$

The corresponding soft Bellman backup is:

$$Q^{\pi}(s_t, a_t) = r(s_t, a_t) + \gamma \, \mathbb{E}_{s_{t+1}}\!\left[V^{\pi}(s_{t+1})\right]$$

$$V^{\pi}(s_t) = \mathbb{E}_{a_t \sim \pi}\!\left[Q^{\pi}(s_t, a_t) - \alpha \log \pi(a_t|s_t)\right]$$

The soft value function subtracts the log-probability (negative entropy contribution), so the optimal soft policy maximizes return while keeping entropy high.

---


## SAC Architecture

SAC uses three neural networks:

1. **Two Q-networks** $Q_{\phi_1}(s,a)$ and $Q_{\phi_2}(s,a)$: estimate state-action values
2. **Policy network** $\pi_\psi(a|s)$: parameterizes a squashed Gaussian for continuous actions

### Why Two Q-Networks?

Double Q-learning (van Hasselt et al., 2016) uses two independent Q-networks and takes the minimum of their predictions to form the Bellman target. This combats **overestimation bias** — a single Q-network systematically overestimates Q-values because the Bellman backup uses the same network to both generate and evaluate actions.

$$Q_{target}(s_t, a_t) = r_t + \gamma \left[\min_{i=1,2} Q_{\phi_i^-}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\psi(a_{t+1}|s_{t+1})\right]$$

where $\phi_i^-$ denotes the **target network** parameters (an exponential moving average of $\phi_i$, updated slowly) and $a_{t+1} \sim \pi_\psi(\cdot|s_{t+1})$.

---

## The Reparameterization Trick

SAC must compute gradients through sampled actions $a \sim \pi_\psi(\cdot|s)$. Direct sampling is not differentiable. The **reparameterization trick** resolves this:

Instead of sampling $a \sim \mathcal{N}(\mu_\psi(s), \sigma_\psi(s))$ directly, sample a noise variable $\xi \sim \mathcal{N}(0, I)$ and compute:

$$a = \tanh\!\left(\mu_\psi(s) + \sigma_\psi(s) \odot \xi\right)$$

The $\tanh$ squashing maps unbounded Gaussian samples to bounded actions in $(-1, 1)^d$. The gradient $\frac{\partial a}{\partial \psi}$ exists and can be backpropagated through $\mu_\psi$ and $\sigma_\psi$.

The log-probability of the squashed action requires a Jacobian correction:

$$\log \pi_\psi(a|s) = \log \mathcal{N}(\tilde{a}; \mu_\psi(s), \sigma_\psi(s)) - \sum_d \log(1 - \tanh^2(\tilde{a}_d))$$

where $\tilde{a}$ is the pre-squash action.

---

## Automatic Temperature Tuning

Choosing $\alpha$ manually is difficult: the right entropy level depends on the task. Haarnoja et al. (2018) propose **automatic temperature tuning** by treating $\alpha$ as a dual variable in a constrained optimization:

$$\max_\alpha \; \mathbb{E}_{s_t, a_t \sim \mathcal{D}}\!\left[-\alpha \log \pi_\psi(a_t|s_t) - \alpha \bar{\mathcal{H}}\right]$$

where $\bar{\mathcal{H}}$ is a **target entropy** (a hyperparameter), typically set to $-\dim(\mathcal{A})$ for continuous action spaces.

This means: if the policy entropy is above the target, decrease $\alpha$ (less entropy regularization needed). If below, increase $\alpha$ (push the policy to explore more).

---

## Full SAC Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ── Network Definitions ────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """
    Critic: estimates Q(s, a).
    Input: concatenated (state, action). Output: scalar Q-value.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),         nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """
    Stochastic actor: parameterizes a squashed Gaussian.
    Outputs mean and log-std; samples via reparameterization.
    """
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, state):
        h = self.trunk(state)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Reparameterized sample from squashed Gaussian.

        Returns:
            action:   tanh-squashed action in (-1, 1)^act_dim
            log_prob: log probability of the action (with Jacobian correction)
            mean:     deterministic action (tanh of mean parameter)
        """
        mean, log_std = self(state)
        std = log_std.exp()

        # Reparameterization: action = tanh(mean + std * noise)
        noise  = torch.randn_like(mean)
        x_t    = mean + std * noise          # pre-squash sample
        action = torch.tanh(x_t)             # squash to (-1, 1)

        # Log-probability with Jacobian correction for tanh squashing
        log_prob = (
            -0.5 * ((x_t - mean) / (std + 1e-6)) ** 2
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=-1)
        # Jacobian: d(tanh(x))/dx = 1 - tanh^2(x)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob, torch.tanh(mean)


# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer for off-policy experience storage."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(np.stack(state)),
            torch.FloatTensor(np.stack(action)),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(np.stack(next_state)),
            torch.FloatTensor(done).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ── SAC Agent ─────────────────────────────────────────────────────────────────

class SACAgent:
    """
    Soft Actor-Critic agent (Haarnoja et al., 2018).

    Features:
    - Double Q-networks for overestimation control
    - Soft target networks (Polyak averaging)
    - Automatic temperature (alpha) tuning
    - Reparameterized Gaussian policy with tanh squashing
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 target_entropy: float = None, alpha_init: float = 0.2,
                 buffer_capacity: int = 1_000_000, batch_size: int = 256):

        self.gamma = gamma
        self.tau   = tau                # Polyak averaging coefficient
        self.batch_size = batch_size

        # ── Critics: two Q-networks + their target copies ──
        self.q1        = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q2        = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dim)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dim)
        # Initialize targets identical to online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ── Actor ──
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim)

        # ── Automatic temperature tuning ──
        self.target_entropy = target_entropy if target_entropy else -act_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp().item()

        # ── Optimizers ──
        self.q1_optimizer     = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer     = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.alpha_optimizer  = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state, deterministic: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.policy.sample(state_t)
        if deterministic:
            return mean.squeeze(0).numpy()
        return action.squeeze(0).numpy()

    def update(self):
        """One gradient step on all networks using a replay buffer sample."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # ── Compute soft Bellman target ──────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            # Min of two targets — overestimation control
            q_next  = torch.min(q1_next, q2_next)
            # Soft Bellman backup: subtract entropy term alpha * log_prob
            target_q = rewards + (1 - dones) * self.gamma * (
                q_next - self.alpha * next_log_probs.unsqueeze(1)
            )

        # ── Update Q-networks ────────────────────────────────────────────────
        q1_loss = F.mse_loss(self.q1(states, actions).unsqueeze(1), target_q)
        q2_loss = F.mse_loss(self.q2(states, actions).unsqueeze(1), target_q)

        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()

        # ── Update policy ────────────────────────────────────────────────────
        actions_new, log_probs, _ = self.policy.sample(states)
        q1_new = self.q1(states, actions_new)
        q2_new = self.q2(states, actions_new)
        q_new  = torch.min(q1_new, q2_new)

        # Policy loss: maximize Q - alpha * log_prob (minimize negative)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ── Update temperature alpha ─────────────────────────────────────────
        alpha_loss = -(self.log_alpha.exp() * (
            log_probs.detach() + self.target_entropy
        )).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().item()

        # ── Soft update target networks (Polyak averaging) ───────────────────
        for param, target_param in zip(self.q1.parameters(),
                                       self.q1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(self.q2.parameters(),
                                       self.q2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "q1_loss":      q1_loss.item(),
            "q2_loss":      q2_loss.item(),
            "policy_loss":  policy_loss.item(),
            "alpha_loss":   alpha_loss.item(),
            "alpha":        self.alpha,
            "entropy":     -log_probs.mean().item(),
        }
```
</div>

---

## Diagram: SAC Data Flow

```
                     ┌─────────────────────────────────────┐
                     │          REPLAY BUFFER              │
                     │  (s, a, r, s', done) × 1M           │
                     └──────────────┬──────────────────────┘
                                    │ sample batch
              ┌──────────────────── ▼ ─────────────────────────┐
              │                                                 │
    ┌─────────┴────────┐                          ┌────────────┴───────┐
    │  Q-networks       │                          │  Policy network    │
    │  Q₁(s,a)  Q₂(s,a)│   ← min →  Q_target      │  π_ψ(a|s)         │
    │  Q₁-target        │                          │  Gaussian + tanh   │
    │  Q₂-target        │                          └────────────────────┘
    └──────────────────-┘                                    │
             │                                               │ reparameterize
             │ soft Bellman target:                          │ a = tanh(μ + σε)
             │ r + γ(min Q_target - α log π)                │
             │                                               │
             └──────────── Q-loss ◄────────────── policy gradient
                           (MSE)                  (max Q - α log π)
                                                            │
                                                   alpha update:
                                            min α * (log π + H_target)
```

---

## SAC vs PPO: When to Use Which

| Dimension | PPO | SAC |
|---|---|---|
| Policy | On-policy | Off-policy |
| Data efficiency | Low (discards data after update) | High (replays data many times) |
| Action space | Discrete or continuous | Continuous primarily |
| Convergence speed | Slower (sample-wise) | Faster (sample-wise) |
| Implementation complexity | Low | Medium |
| Hyperparameter sensitivity | Low | Medium |
| Exploration mechanism | Entropy bonus + stochastic policy | Maximum entropy framework (principled) |
| Parallelism | Scales well with parallel envs | Limited by replay buffer sequential writes |
| RLHF / LLM fine-tuning | Dominant choice | Not standard |

**Use SAC when:**
- Action space is continuous (robotics, control, locomotion)
- Sample efficiency matters (real-robot experiments, expensive simulations)
- You want automatic exploration without tuning entropy coefficients manually

**Use PPO when:**
- Action space is discrete (Atari, game playing)
- Parallel environment rollouts are available
- Problem is episodic with moderate episode length
- You need to fine-tune language models (PPO is the RLHF standard)

---


<div class="compare">
<div class="compare-card">
<div class="header before">SAC</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">PPO: When to Use Which</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Common Pitfalls

<div class="callout-danger">
<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.
</div>

**Pitfall 1 — Replay buffer too small.**
SAC requires a large replay buffer (at least 100K transitions, ideally 1M) for the off-policy updates to be effective. A buffer that is too small causes overfitting to recent experience and instability. This is the most common SAC failure mode.

<div class="callout-warning">
<strong>Warning:</strong> **Pitfall 1 — Replay buffer too small.**
SAC requires a large replay buffer (at least 100K transitions, ideally 1M) for the off-policy updates to be effective.
</div>

**Pitfall 2 — Target entropy set incorrectly.**
The default $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ works well for normalized action spaces. If the action space is not in $[-1, 1]^d$, scale accordingly. Too-high target entropy prevents the policy from committing to good actions; too-low collapses exploration.

**Pitfall 3 — Forgetting the Jacobian correction in log-probability.**
The $\tanh$ squashing changes the probability density. The term $-\sum_d \log(1 - \tanh^2(\tilde{a}_d))$ must be subtracted from the Gaussian log-probability. Omitting this makes the log-prob incorrect and destabilizes the temperature update.

**Pitfall 4 — Action scaling not applied to environment.**
SAC outputs actions in $(-1, 1)^d$ due to $\tanh$ squashing. Most environments expect actions in a different range. Always rescale: `action_env = action_low + 0.5 * (action + 1) * (action_high - action_low)`.

**Pitfall 5 — Updating before the buffer has sufficient data.**
Start gradient updates only after collecting at least `batch_size` (or better, 1000) transitions. Updating on tiny buffers leads to severe overfitting to early experience.

**Pitfall 6 — Target networks not updated slowly enough.**
Polyak coefficient $\tau = 0.005$ (i.e., 99.5% old, 0.5% new per step) is the standard. Using $\tau = 0.1$ or higher makes targets unstable and causes Q-value divergence.

---

## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

- **Builds on:** Actor-critic architectures (Module 6), double Q-learning (Module 5), exploration-exploitation trade-offs (Modules 2-3)
- **Leads to:** TD3 (Fujimoto 2018) adds deterministic policy gradient to the SAC framework; REDQ (Chen 2021) pushes sample efficiency further with ensembles
- **Related to:** MaxEnt IRL (Ziebart 2008) uses the same entropy-regularized objective for inverse RL; variational inference objectives have the same form as the soft Bellman equations

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML 2018. — Primary reference for this guide.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic Algorithms and Applications.* arXiv:1812.05905. — Extended version with automatic temperature tuning; the definitive SAC reference.
- Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. — TD3, the deterministic counterpart to SAC; introduces double Q-networks in the actor-critic setting.
- van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. — Original double Q-learning paper; motivation for SAC's dual-critic design.
- Stable-Baselines3 SAC implementation: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html — production-quality reference implementation with extensive documentation.


---

## Cross-References

<a class="link-card" href="./03_sac_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_ppo_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>

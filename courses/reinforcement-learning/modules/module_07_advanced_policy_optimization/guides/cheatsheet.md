# Module 07 Cheatsheet: Advanced Policy Optimization

## Algorithm Objectives

### TRPO (Schulman et al., 2015)

**Objective (maximize):**

$$L(\theta) = \mathbb{E}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a)\right]$$

**Constraint:**

$$\mathbb{E}\!\left[D_{KL}\!\left(\pi_{\theta_{old}} \,\|\, \pi_\theta\right)\right] \leq \delta$$

**Update rule (natural gradient):**

$$\theta \leftarrow \theta + \alpha \, F(\theta)^{-1} \nabla_\theta J(\theta)$$

where $F(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta \cdot \nabla_\theta \log \pi_\theta^\top\right]$ is the Fisher information matrix.

---

### PPO-Clip (Schulman et al., 2017)

**Probability ratio:**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**PPO-Clip objective (maximize):**

$$L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\right)\right]$$

**Full objective (including value and entropy terms):**

$$L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1 (V_\theta(s_t) - V_t^{targ})^2 + c_2 \mathcal{H}[\pi_\theta](\cdot|s_t)$$

---

### SAC (Haarnoja et al., 2018)

**Maximum entropy objective (maximize):**

$$J(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

**Soft Bellman target (for Q-network updates):**

$$Q_{target}(s_t, a_t) = r_t + \gamma \left[\min_{i=1,2} Q_{\phi_i^-}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\psi(a_{t+1}|s_{t+1})\right]$$

**Policy update (maximize):**

$$J(\psi) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\psi}\!\left[Q_\phi(s, a) - \alpha \log \pi_\psi(a|s)\right]$$

**Automatic temperature update (minimize):**

$$J(\alpha) = \mathbb{E}_{a \sim \pi_\psi}\!\left[-\alpha \log \pi_\psi(a|s) - \alpha \bar{\mathcal{H}}\right]$$

---

## Notation Reference

| Symbol | Meaning |
|---|---|
| $\pi_\theta(a \mid s)$ | Policy: probability of action $a$ in state $s$ under parameters $\theta$ |
| $\pi_{\theta_{old}}$ | Policy from which data was collected (fixed during update) |
| $\hat{A}(s,a)$ | Advantage estimate: how much better $a$ is than average at $s$ |
| $r_t(\theta)$ | Importance ratio: $\pi_\theta / \pi_{\theta_{old}}$ |
| $\epsilon$ | PPO clip parameter (typically 0.2) |
| $\delta$ | TRPO KL constraint radius (typically 0.01) |
| $F(\theta)$ | Fisher information matrix |
| $\alpha$ | Temperature (entropy coefficient) in SAC |
| $\mathcal{H}(\pi)$ | Shannon entropy: $-\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ |
| $\bar{\mathcal{H}}$ | Target entropy in SAC (typically $-\dim(\mathcal{A})$) |
| $\tau$ | Polyak averaging coefficient for target networks (SAC; typically 0.005) |
| $\phi^-$ | Target network parameters (slow-moving copy of $\phi$) |
| $c_1$ | Value loss coefficient in PPO (typically 0.5) |
| $c_2$ | Entropy bonus coefficient in PPO (typically 0.01) |
| $\mathcal{D}$ | Replay buffer (off-policy) |

---

## Algorithm Comparison Table

| Dimension | TRPO | PPO | SAC |
|---|---|---|---|
| **Reference** | Schulman 2015 | Schulman 2017 | Haarnoja 2018 |
| **Policy type** | Stochastic | Stochastic | Stochastic |
| **Data usage** | On-policy | On-policy | Off-policy |
| **Constraint mechanism** | Hard KL constraint | Clipped ratio | Entropy bonus |
| **Optimizer** | CG + line search | Adam | Adam |
| **Trust region** | Exact (KL $\leq \delta$) | Approximate (clip) | None (objective-level) |
| **Action space** | Discrete or continuous | Discrete or continuous | Continuous (primary) |
| **Sample efficiency** | Low | Low | High |
| **Implementation complexity** | High | Low | Medium |
| **Parallel environments** | Benefits significantly | Benefits significantly | Limited benefit |
| **Monotonic improvement** | Theoretical guarantee | No | No |
| **Exploration mechanism** | Stochastic policy + GAE | Entropy bonus | Max entropy objective |
| **Double Q-networks** | No | No | Yes |
| **Replay buffer** | No | No | Yes (1M transitions) |
| **Hyperparameter sensitivity** | Medium | Low | Medium |

---

## Hyperparameter Ranges

### TRPO

| Parameter | Typical Range | Default |
|---|---|---|
| KL constraint $\delta$ | $0.005 {-} 0.05$ | $0.01$ |
| Discount $\gamma$ | $0.99 {-} 0.999$ | $0.99$ |
| GAE $\lambda$ | $0.9 {-} 0.99$ | $0.97$ |
| CG iterations | $10 {-} 50$ | $10$ |
| Backtrack coefficient | $0.5 {-} 0.9$ | $0.8$ |
| Max backtrack steps | $10 {-} 20$ | $10$ |
| Advantage normalization | Always | — |

### PPO

| Parameter | Typical Range | Default |
|---|---|---|
| Clip $\epsilon$ | $0.1 {-} 0.3$ | $0.2$ |
| Discount $\gamma$ | $0.99 {-} 0.999$ | $0.99$ |
| GAE $\lambda$ | $0.9 {-} 0.99$ | $0.95$ |
| Learning rate | $10^{-4} {-} 3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| Gradient clip norm | $0.5 {-} 1.0$ | $0.5$ |
| Epochs per rollout | $3 {-} 10$ | $4$ |
| Mini-batch size | $64 {-} 2048$ | $64$ |
| Value loss coefficient $c_1$ | $0.25 {-} 1.0$ | $0.5$ |
| Entropy coefficient $c_2$ | $0.001 {-} 0.05$ | $0.01$ |
| Advantage normalization | Always | — |

### SAC

| Parameter | Typical Range | Default |
|---|---|---|
| Learning rate (all networks) | $10^{-4} {-} 3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| Discount $\gamma$ | $0.99 {-} 0.999$ | $0.99$ |
| Polyak $\tau$ | $0.001 {-} 0.01$ | $0.005$ |
| Batch size | $128 {-} 512$ | $256$ |
| Replay buffer size | $100\text{K} {-} 1\text{M}$ | $1\text{M}$ |
| Warm-up steps | $1\text{K} {-} 10\text{K}$ | $1\text{K}$ |
| Target entropy $\bar{\mathcal{H}}$ | $-\dim(\mathcal{A})$ | $-\dim(\mathcal{A})$ |
| Initial temperature $\alpha$ | $0.1 {-} 1.0$ | $0.2$ (then auto-tuned) |
| Hidden units | $256 {-} 512$ | $256$ |
| Log-std range | $[-5, 2]$ | $[-5, 2]$ |

---

## Decision Guide: Which Algorithm to Use

```
START
  │
  ├─ Is the action space DISCRETE?
  │      YES → Use PPO (SAC not designed for discrete; TRPO possible but slow)
  │
  ├─ Do you need a PROVABLE safety guarantee (monotonic improvement)?
  │      YES → Use TRPO (only algorithm with theoretical backing)
  │
  ├─ Is SAMPLE EFFICIENCY critical?
  │   (real hardware, expensive simulation, limited data budget)
  │      YES → Use SAC (off-policy, replays each transition many times)
  │
  ├─ Are MANY PARALLEL ENVIRONMENTS available?
  │   (vectorized gym, HPC cluster)
  │      YES → Use PPO (scales linearly with parallel envs; SAC does not)
  │
  ├─ Is IMPLEMENTATION SIMPLICITY a priority?
  │   (teaching, quick prototype, time-constrained)
  │      YES → Use PPO (~100 lines; SAC ~300; TRPO ~400)
  │
  ├─ Fine-tuning a LANGUAGE MODEL (RLHF)?
  │      YES → Use PPO (established standard; SAC not used in LLM context)
  │
  └─ CONTINUOUS CONTROL benchmark (MuJoCo, dm_control)?
         → Use SAC (state of the art; reaches better asymptotic performance)
           with PPO as baseline for comparison
```

---

## Common Diagnostic Signals

### PPO Diagnostics

| Metric | Healthy Range | Action if Unhealthy |
|---|---|---|
| Approx KL divergence | $< 0.02$ | Reduce LR or $\epsilon$ if $> 0.05$ |
| Clip fraction | $0.1 {-} 0.3$ | Reduce $\epsilon$ if $> 0.5$ |
| Policy entropy | Slow decrease | Increase $c_2$ if collapses |
| Value loss | Decreasing | Increase $c_1$ or separate optimizer |
| Explained variance | $> 0.5$ | Improve value function capacity |

### SAC Diagnostics

| Metric | Healthy Range | Action if Unhealthy |
|---|---|---|
| Q-loss (both critics) | Decreasing | Check replay buffer size, target $\tau$ |
| Policy loss | Negative, decreasing | Check log-prob computation (tanh Jacobian) |
| Temperature $\alpha$ | Converges in $[0.05, 0.5]$ | Check target entropy sign |
| Policy entropy | Near $\bar{\mathcal{H}}$ | Adjust $\bar{\mathcal{H}}$ if action space non-standard |
| Q-value magnitude | Bounded, not exploding | Reduce $\tau$ or LR if diverging |

---

## Quick-Reference Formulas

**GAE (used by TRPO and PPO):**

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**KL divergence (TRPO constraint direction):**

$$D_{KL}(\pi_{old} \| \pi_{new}) = \mathbb{E}_{a \sim \pi_{old}}\!\left[\log \frac{\pi_{old}(a|s)}{\pi_{new}(a|s)}\right]$$

**Reparameterized action (SAC):**

$$a = \tanh(\mu_\psi(s) + \sigma_\psi(s) \odot \xi), \quad \xi \sim \mathcal{N}(0, I)$$

**Log-probability with tanh correction (SAC):**

$$\log \pi(a|s) = \log \mathcal{N}(\tilde{a}; \mu, \sigma) - \sum_d \log(1 - \tanh^2(\tilde{a}_d))$$

**Polyak target update (SAC):**

$$\phi^- \leftarrow \tau \phi + (1-\tau)\phi^-$$

**Advantage normalization (PPO and TRPO):**

$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \text{mean}(\hat{A})}{\text{std}(\hat{A}) + \epsilon}, \quad \epsilon = 10^{-8}$$

# Module 06 — Policy Gradient Methods: Cheatsheet

> **Reading time:** ~5 min | **Module:** 6 — Policy Gradient | **Prerequisites:** Module 5

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Policy (actor) network parameters |
| $\mathbf{w}$ | Value (critic) network parameters |
| $\pi(a\|s;\theta)$ | Policy: probability of action $a$ in state $s$ |
| $V(s;\mathbf{w})$ | Critic: estimated state value under current policy |
| $Q^{\pi}(s,a)$ | Action-value: expected return from $(s,a)$ following $\pi$ |
| $A^{\pi}(s,a)$ | Advantage: $Q^{\pi}(s,a) - V^{\pi}(s)$ |
| $G_t$ | Discounted return from time $t$ |
| $\delta_t$ | One-step TD error (advantage estimate) |
| $\gamma$ | Discount factor $\in [0,1)$ |
| $\lambda$ | GAE interpolation parameter $\in [0,1]$ |
| $\alpha_\theta, \alpha_w$ | Actor and critic learning rates |

---

## 1. Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(A|S) \cdot Q^{\pi_\theta}(S,A)\right]$$

<div class="callout-insight">
<strong>Insight:</strong> $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(A|S) \cdot Q^{\pi_\theta}(S,A)\right]$$

**Key facts:**
- Environment dynamics $p(s'|s,a)$ do not appear — mode...
</div>


**Key facts:**
- Environment dynamics $p(s'|s,a)$ do not appear — model-free
- Estimate by sampling trajectories under $\pi_\theta$ (on-policy)
- $\nabla_\theta \log \pi_\theta(a|s)$ (score function) is computed by autograd

> Reference: Sutton & Barto, *RL: An Introduction* (2nd ed.), Chapter 13; Sutton et al. (2000) NeurIPS

---

## 2. REINFORCE Update Rules

### Vanilla REINFORCE

<div class="callout-key">
<strong>Key Point:</strong> ### Vanilla REINFORCE

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

where the return from time $t$ is:
$$G_t = \sum_{k=0}^{T-t-1} \gamma^...
</div>


$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

where the return from time $t$ is:
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T$$

**Properties:** Unbiased | High variance | Requires complete episodes

### REINFORCE with Baseline

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} \gamma^t (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t|S_t)$$

The baseline $b(S_t)$ must not depend on $A_t$. It does not introduce bias:

$$\mathbb{E}_{A_t}\!\left[b(S_t)\nabla_\theta \log \pi_\theta(A_t|S_t)\right] = 0$$

**Common baseline choices:**

| Baseline | Complexity | Variance Reduction |
|----------|------------|-------------------|
| $b = 0$ | None | None |
| $b = \bar{G}$ (running average) | Trivial | Moderate |
| $b(s) = V^{\pi}(s)$ (learned) | Critic network | Best |

> Reference: Williams (1992). *Machine Learning* 8(3-4), 229-256

---

## 3. Advantage Function

$$A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$$

<div class="callout-info">
<strong>Info:</strong> $$A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$$

**Interpretation:** How much better (or worse) is action $a$ compared to the average action in state $s$ under policy $\pi_\theta$?
</div>


**Interpretation:** How much better (or worse) is action $a$ compared to the average action in state $s$ under policy $\pi_\theta$?

**Properties:**
- $\mathbb{E}_{a \sim \pi}[A^{\pi}(s,a)] = 0$ (zero mean under policy)
- $A > 0$: action is better than average — increase its probability
- $A < 0$: action is worse than average — decrease its probability

---

## 4. Actor-Critic Update Rules

**TD error (one-step advantage estimate):**
$$\delta_t = R_{t+1} + \gamma V(S_{t+1};\mathbf{w}) - V(S_t;\mathbf{w})$$

<div class="callout-warning">
<strong>Warning:</strong> **TD error (one-step advantage estimate):**
$$\delta_t = R_{t+1} + \gamma V(S_{t+1};\mathbf{w}) - V(S_t;\mathbf{w})$$

At episode termination: $\delta_t = R_{t+1} - V(S_t;\mathbf{w})$ (zero bootstrap)...
</div>


At episode termination: $\delta_t = R_{t+1} - V(S_t;\mathbf{w})$ (zero bootstrap)

**Critic update** (minimize squared TD error):
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha_w\, \delta_t\, \nabla_{\mathbf{w}} V(S_t;\mathbf{w})$$

**Actor update** (policy gradient with TD advantage):
$$\theta \leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \log \pi(A_t|S_t;\theta)$$

**Structural rules:**
- Actor and critic are **separate networks** with separate parameters $\theta$ and $\mathbf{w}$
- Use **separate optimizers** with different learning rates ($\alpha_w > \alpha_\theta$ typically)
- Use `td_error.detach()` when computing the actor loss (prevent gradient cross-contamination)
- Zero bootstrap at terminal states: multiply $V(S_{t+1})$ by $(1 - \text{done})$

> Reference: Sutton & Barto, Ch. 13.5; Mnih et al. (2016) ICML (A3C/A2C)

---

## 5. Generalized Advantage Estimation (GAE)

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

**Efficient backward recurrence:**
$$\hat{A}_{T-1} = \delta_{T-1}, \qquad \hat{A}_t = \delta_t + \gamma\lambda\,(1-d_{t+1})\,\hat{A}_{t+1}$$

**$\lambda$ controls bias-variance tradeoff:**

| $\lambda$ | Equivalent | Bias | Variance |
|-----------|-----------|------|----------|
| $0$ | 1-step TD: $\hat{A}_t = \delta_t$ | High (critic error) | Low |
| $1$ | Monte Carlo: $\hat{A}_t = G_t - V(S_t)$ | Zero | High |
| $0.95$ | Weighted multi-step (default in PPO) | Low | Moderate |

> Reference: Schulman et al. (2016). *ICLR*. High-dimensional continuous control using GAE

---

## 6. Policy Parameterizations

### Softmax Policy (Discrete Actions)

$$\pi_\theta(a|s) = \frac{\exp(\theta^\top \phi(s,a))}{\sum_{a'} \exp(\theta^\top \phi(s,a'))}$$

**Score function:**
$$\nabla_\theta \log \pi_\theta(a|s) = \phi(s,a) - \mathbb{E}_{\pi_\theta}[\phi(s,\cdot)]$$

### Gaussian Policy (Continuous Actions)

$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s),\, \sigma^2_\theta(s))$$

**Score functions:**
$$\nabla_{\theta_\mu} \log \pi_\theta(a|s) = \frac{a - \mu_\theta(s)}{\sigma^2_\theta(s)}\,\phi(s)$$

$$\nabla_{\theta_\sigma} \log \pi_\theta(a|s) = \left(\frac{(a - \mu_\theta(s))^2}{\sigma^2_\theta(s)} - 1\right)\phi(s)$$

### Comparison

| Property | Softmax | Gaussian |
|----------|---------|----------|
| Action space | Discrete $\{1,\ldots,K\}$ | Continuous $\mathbb{R}^d$ |
| Output | Probabilities over $K$ actions | Mean $\mu(s)$ and std $\sigma(s)$ |
| Exploration | Controlled by logit scale | Controlled by $\sigma(s)$ |
| Common use | Discrete control, board games | Robotics, continuous control |

---

## 7. Policy Gradient vs Value-Based: Comparison

| Dimension | Policy Gradient | Value-Based (DQN) |
|-----------|----------------|-------------------|
| **What is learned** | Policy $\pi(a\|s;\theta)$ directly | Q-function $Q(s,a;\mathbf{w})$ |
| **Action selection** | Sample from $\pi(\cdot\|s)$ | $\arg\max_a Q(s,a)$ |
| **Action spaces** | Discrete and continuous | Discrete (naturally) |
| **Stochastic policies** | Native support | Requires $\epsilon$-greedy |
| **Convergence** | To local optimum (with FA) | May diverge (deadly triad) |
| **Sample efficiency** | Lower (on-policy) | Higher (replay buffer) |
| **Variance** | High (mitigated by baselines) | Lower (TD target) |
| **Primary reference** | Sutton et al. 2000; Williams 1992 | Mnih et al. 2015 (DQN) |
| **Modern successor** | PPO, SAC | Rainbow DQN, C51 |

**Key rule:** Use policy gradient when you need stochastic policies, continuous action spaces, or convergence guarantees. Use value-based when data efficiency and off-policy learning matter most.

---

## Key References

| Paper | Contribution |
|-------|-------------|
| Williams (1992). *Machine Learning* | REINFORCE algorithm; score function estimator |
| Sutton et al. (2000). *NeurIPS* | Policy gradient theorem; eliminates dynamics from gradient |
| Mnih et al. (2016). *ICML* | A3C and A2C; asynchronous parallel actor-critic |
| Schulman et al. (2016). *ICLR* | GAE; bias-variance tradeoff in advantage estimation |
| Sutton & Barto (2018). Ch. 13 | Comprehensive treatment of policy gradient methods |

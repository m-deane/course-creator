# Module 05 Deep RL — Cheatsheet

> **Reading time:** ~5 min | **Module:** 5 — Deep RL | **Prerequisites:** Module 4, PyTorch basics

## DQN Loss Function

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,\, a,\, r,\, s',\, \text{done}) \sim \mathcal{D}}\!\left[\bigl(Y - Q(s, a;\, \theta)\bigr)^2\right]$$

<div class="callout-insight">
<strong>Insight:</strong> $$\mathcal{L}(\theta) = \mathbb{E}_{(s,\, a,\, r,\, s',\, \text{done}) \sim \mathcal{D}}\!\left[\bigl(Y - Q(s, a;\, \theta)\bigr)^2\right]$$

$$Y = \begin{cases} r & \text{if done} \\ r + \gamma \disp...
</div>


$$Y = \begin{cases} r & \text{if done} \\ r + \gamma \displaystyle\max_{a'} Q(s',\, a';\, \theta^-) & \text{otherwise} \end{cases}$$

- $\theta$: online (trained) Q-network parameters
- $\theta^-$: target network parameters (frozen, synced every $C$ steps)
- $\mathcal{D}$: replay buffer

---

## Double DQN Target

**Decouple action selection from action evaluation:**

<div class="callout-key">
<strong>Key Point:</strong> **Decouple action selection from action evaluation:**

$$Y^{\text{DDQN}} = r + \gamma\, Q\!\left(s',\; \underbrace{\arg\max_a Q(s', a;\, \theta)}_{\text{selection by } \theta},\; \underbrace{\theta^-}...
</div>


$$Y^{\text{DDQN}} = r + \gamma\, Q\!\left(s',\; \underbrace{\arg\max_a Q(s', a;\, \theta)}_{\text{selection by } \theta},\; \underbrace{\theta^-}_{\text{evaluation}}\right)$$

**Comparison:**

| | Action selection | Action evaluation |
|-|-----------------|------------------|
| DQN | $\theta^-$ | $\theta^-$ |
| Double DQN | $\theta$ (online) | $\theta^-$ (target) |

Motivation: $\mathbb{E}[\max_a Q(s', a; \theta^-)] \geq \max_a Q^*(s', a)$ — max of noisy estimates is biased upward.

---

## Dueling Network Architecture

$$Q(s, a;\, \theta) = V(s;\, \theta_V) + \left(A(s, a;\, \theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\, \theta_A)\right)$$

<div class="callout-info">
<strong>Info:</strong> $$Q(s, a;\, \theta) = V(s;\, \theta_V) + \left(A(s, a;\, \theta_A) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a';\, \theta_A)\right)$$

- $V(s; \theta_V) \in \mathbb{R}$: state-value stream (scalar)
- $A...
</div>


- $V(s; \theta_V) \in \mathbb{R}$: state-value stream (scalar)
- $A(s, a; \theta_A) \in \mathbb{R}^{|\mathcal{A}|}$: advantage stream (one per action)
- Mean subtraction enforces identifiability: $V(s)$ is the true state value

```
State s → Shared Encoder → Value head    → V(s)    ─┐
                         → Advantage head → A(s,·) ─→ Q = V + (A − mean(A))
```

---

## PER Priority Formula

**Priority** (proportional variant):

<div class="callout-warning">
<strong>Warning:</strong> **Priority** (proportional variant):

$$p_i = |\delta_i| + \epsilon \qquad \delta_i = Y_i - Q(s_i, a_i;\theta), \quad \epsilon = 0.01$$

**Sampling probability:**

$$P(i) = \frac{p_i^\alpha}{\sum_k p_...
</div>


$$p_i = |\delta_i| + \epsilon \qquad \delta_i = Y_i - Q(s_i, a_i;\theta), \quad \epsilon = 0.01$$

**Sampling probability:**

$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha} \qquad \alpha = 0.6 \text{ (typical)}$$

**Importance sampling weight** (bias correction):

$$w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta, \quad \text{normalized by } \max_j w_j$$

$\beta$ annealed from $\beta_0 = 0.4$ to $1.0$ over training.

**Weighted loss:**

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[w_i \cdot (Y_i - Q(s_i, a_i;\theta))^2\right]$$

---

## Hyperparameter Ranges

| Hyperparameter | Typical Range | Recommended Start |
|----------------|--------------|-------------------|
| Learning rate $\alpha$ | $10^{-5}$ – $10^{-3}$ | $10^{-4}$ (Adam) |
| Replay buffer size $N$ | 10K – 1M | 100K |
| Batch size | 32 – 256 | 64 |
| Discount $\gamma$ | 0.9 – 0.999 | 0.99 |
| Target update freq $C$ | 100 – 10K steps | 1,000 |
| $\epsilon$ start | 1.0 | 1.0 (always) |
| $\epsilon$ end | 0.01 – 0.1 | 0.01 |
| $\epsilon$ decay steps | 10K – 1M | 100K |
| Gradient clip max norm | 1 – 10 | 10 |
| PER $\alpha$ exponent | 0.4 – 0.8 | 0.6 |
| PER $\beta_0$ start | 0.3 – 0.6 | 0.4 |
| Training start threshold | $\geq$ batch size × 10 | 1K – 50K |

---

## Debugging Checklist

### Before Training
- [ ] Target network initialized to same weights as Q-network
- [ ] Target network is NOT in `optimizer.param_groups` (no gradient updates)
- [ ] Replay buffer capacity set to at least 10× episode length
- [ ] Buffer warmup threshold ≥ `batch_size × 10`
- [ ] `torch.no_grad()` wraps all target network forward passes
- [ ] Terminal state masking: multiply by `(1.0 - dones)`
- [ ] `set_global_seed(seed)` called before any initialization
- [ ] Gradient clipping enabled: `clip_grad_norm_(params, max_norm=10.0)`

### During Training — Log Every Update Step
- [ ] TD loss: should decrease (eventually); NaN = divergence
- [ ] Mean Q-value: should grow slowly; unbounded growth = overestimation
- [ ] Gradient L2 norm: should stay below ~10; spikes = instability
- [ ] Epsilon: should decay on schedule

### Evaluation
- [ ] Evaluation uses $\epsilon = 0$ (greedy), NOT training epsilon
- [ ] Separate `eval_env` from `train_env`
- [ ] Results reported over ≥ 5 independent seeds
- [ ] Report IQM or median, not just mean

### Reproducibility
- [ ] Random seeds set: `random`, `numpy`, `torch`, environment
- [ ] `torch.backends.cudnn.deterministic = True`
- [ ] Library versions recorded (`torch`, `gymnasium`, `numpy`)
- [ ] Checkpoints saved every 100K steps

---

## Algorithm Comparison

| | DQN | Double DQN | Dueling | +PER | Rainbow |
|-|-----|-----------|---------|------|---------|
| Overestimation | High | Low | Moderate | High | Low |
| Architecture | Standard | Standard | $V + A - \bar{A}$ | Standard | Dueling + Noisy |
| Sampling | Uniform | Uniform | Uniform | Prioritized | Prioritized |
| Target | $\max Q(s';\theta^-)$ | $Q(s', a^*_\theta; \theta^-)$ | Same as DQN | Same as DQN | $n$-step Double |
| Code change | Baseline | 2 lines | Network only | Buffer + weights | All of the above |

---

## Key References

| Paper | Year | Contribution |
|-------|------|-------------|
| Mnih et al. (Nature) | 2015 | DQN: experience replay + target network |
| van Hasselt et al. | 2016 | Double DQN: decouple selection from evaluation |
| Wang et al. | 2016 | Dueling DQN: $V + A$ architecture |
| Schaul et al. | 2016 | Prioritized Experience Replay |
| Hessel et al. | 2018 | Rainbow: all improvements combined |
| Henderson et al. | 2018 | Deep RL reproducibility analysis |
| Agarwal et al. | 2021 | IQM and robust RL evaluation |

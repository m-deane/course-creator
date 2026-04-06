# Module 8: Model-Based RL — Cheatsheet

> **Reading time:** ~8 min | **Module:** 8 — Model-Based RL | **Prerequisites:** Module 5

## Model-Free vs Model-Based Comparison

| Dimension | Model-Free | Model-Based |
|-----------|-----------|-------------|
| **Learns** | Value function or policy directly | Environment model $\hat{p}(s'\|s,a)$, $\hat{r}(s,a)$ |
| **Data use** | Each transition used once (or few times) | Transitions also used to train model; model generates additional synthetic transitions |
| **Sample efficiency** | Lower: 10⁵–10⁷ steps typical | Higher: 10³–10⁵ steps typical |
| **Compute per step** | Low | Higher (model training + planning) |
| **Key risk** | High variance, slow convergence | Model exploitation, compounding errors |
| **Requires model** | No | Yes (learned from data) |
| **Representative algorithms** | DQN, PPO, SAC, A3C | Dyna-Q, MCTS, World Models, MuZero, MBPO |
| **Best suited for** | Fast simulators, complex dynamics | Expensive real interactions, learnable dynamics |

---

## Dyna-Q Algorithm Steps

**Input:** $\alpha$ (step size), $\gamma$ (discount), $\epsilon$ (exploration), $n$ (planning steps)

<div class="callout-insight">
<strong>Insight:</strong> **Input:** $\alpha$ (step size), $\gamma$ (discount), $\epsilon$ (exploration), $n$ (planning steps)



**Source:** Sutton & Barto (2018), Algorithm 8.1

**Dyna-Q+ modification (non-stationary environ...
</div>


```
Initialize Q(s, a) = 0 for all s ∈ S, a ∈ A
Initialize Model(s, a) = empty

Repeat for each episode step:
  (1) Observe current state S
  (2) Choose A ← ε-greedy w.r.t. Q(S, ·)
  (3) Execute A; observe R, S'

  --- Direct RL update (real experience) ---
  (4) Q(S, A) ← Q(S, A) + α [R + γ max_a Q(S', a) - Q(S, A)]

  --- Model learning ---
  (5) Model(S, A) ← (R, S')

  --- Planning (n times) ---
  (6) For i = 1 to n:
        S̃ ← random previously observed state
        Ã ← random action previously taken in S̃
        R̃, S̃' ← Model(S̃, Ã)
        Q(S̃, Ã) ← Q(S̃, Ã) + α [R̃ + γ max_a Q(S̃', a) - Q(S̃, Ã)]

  (7) S ← S'
```

**Source:** Sutton & Barto (2018), Algorithm 8.1

**Dyna-Q+ modification (non-stationary environments):**

Replace $R$ with $\tilde{r}(S, A) = R + \kappa\sqrt{\tau(S, A)}$, where $\tau(S,A)$ counts steps since $(S,A)$ was last tried in the real environment and $\kappa = 0.001$–$0.01$.

---

## MCTS Four Phases

### Phase 1: Selection
Starting from root $s_0$, traverse existing tree by selecting the child with the highest UCT score until reaching a **leaf node** (a node with at least one untried action, or a terminal state).

<div class="callout-key">
<strong>Key Point:</strong> ### Phase 1: Selection
Starting from root $s_0$, traverse existing tree by selecting the child with the highest UCT score until reaching a **leaf node** (a node with at least one untried action, or a ...
</div>


### Phase 2: Expansion
At the leaf node, add one new child node corresponding to an untried action. The new node is initialized with $N = 0$, $\bar{Q} = 0$.

### Phase 3: Simulation (Rollout)
From the newly expanded node, run a **rollout** to a terminal state (or depth limit $H$) using the rollout policy $\pi_\text{roll}$ (typically uniform random). Record the cumulative return $G$.

### Phase 4: Backpropagation
Propagate $G$ from the expanded node back to the root, updating each node on the path:
$$N(s) \leftarrow N(s) + 1 \qquad N(s,a) \leftarrow N(s,a) + 1$$
$$\bar{Q}(s,a) \leftarrow \bar{Q}(s,a) + \frac{G - \bar{Q}(s,a)}{N(s,a)}$$

**Final action selection** (after $K$ iterations):
$$a^* = \arg\max_a N(s_0, a) \qquad \text{(visit count, not Q-value)}$$

---

## UCB1 / UCT Formula

**UCB1** (multi-armed bandits):
$$\text{UCB1}(a) = \bar{r}(a) + c\sqrt{\frac{\ln N}{N(a)}}$$

<div class="callout-info">
<strong>Info:</strong> **UCB1** (multi-armed bandits):
$$\text{UCB1}(a) = \bar{r}(a) + c\sqrt{\frac{\ln N}{N(a)}}$$

**UCT** (MCTS tree policy):
$$\text{UCT}(s, a) = \bar{Q}(s, a) + c\sqrt{\frac{\ln N(s)}{N(s, a)}}$$

| Sym...
</div>


**UCT** (MCTS tree policy):
$$\text{UCT}(s, a) = \bar{Q}(s, a) + c\sqrt{\frac{\ln N(s)}{N(s, a)}}$$

| Symbol | Meaning |
|--------|---------|
| $\bar{Q}(s, a)$ | Empirical mean return from $(s, a)$ across all simulations |
| $N(s)$ | Total visit count for state $s$ (parent) |
| $N(s, a)$ | Visit count for action $a$ from state $s$ |
| $c$ | Exploration coefficient; $c = \sqrt{2}$ (theoretical); tune per domain |

**AlphaGo / AlphaZero PUCT variant:**
$$\text{PUCT}(s, a) = \bar{Q}(s, a) + c \cdot p(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

where $p(a \mid s)$ is the policy network prior over actions.

**Behavior of UCT:**
- $N(s,a) = 0$: UCT = $\infty$ → always expand unvisited nodes first
- $N(s,a)$ small: large exploration bonus → probe under-explored actions
- $N(s,a)$ large: $\bar{Q}$ dominates → exploit well-explored actions

---

## MuZero Three Functions

| Function | Formula | Purpose |
|----------|---------|---------|
| **Representation** $h$ | $h_0 = h(o_1, \ldots, o_t)$ | Encode observation history to initial hidden state |
| **Dynamics** $g$ | $r_k, h_k = g(h_{k-1}, a_k)$ | Predict next hidden state and immediate reward |
| **Prediction** $f$ | $p_k, v_k = f(h_k)$ | Predict policy prior and value from any hidden state |

<div class="callout-warning">
<strong>Warning:</strong> $h_0 = h(o_1, \ldots, o_t)$ — encode current history
2.
</div>


**MuZero planning loop** (decision time):
1. $h_0 = h(o_1, \ldots, o_t)$ — encode current history
2. Run $K$ MCTS simulations using $g$ for expansion, $f$ for leaf evaluation
3. $a^* = \arg\max_a N(h_0, a)$ — take action with most MCTS visits

**MuZero training loss** (unrolled $K$ steps):
$$\mathcal{L} = \sum_{k=0}^{K} \bigl[\underbrace{l^r(u_{t+k}, r_k)}_\text{reward} + \underbrace{l^v(z_{t+k}, v_k)}_\text{value} + \underbrace{l^p(\pi_{t+k}, p_k)}_\text{policy}\bigr]$$

where $u_{t+k}$ = actual reward, $z_{t+k}$ = bootstrapped value target, $\pi_{t+k}$ = MCTS search policy.

---

## World Models Three Components

| Component | Function | Training Objective |
|-----------|----------|-------------------|
| **V** — Vision (VAE) | $z_t = \text{encode}(o_t)$, $d_z \ll d_o$ | Reconstruct $o_t$ from $z_t$; minimize ELBO |
| **M** — Memory (MDN-RNN) | $\hat{z}_{t+1} \sim p(z_{t+1} \mid z_t, a_t, h_t)$, $h_{t+1} = \text{LSTM}(\ldots)$ | Maximum likelihood on encoded sequences |
| **C** — Controller | $a_t = W_c [z_t; h_t] + b_c$ | Cumulative dream reward (CMA-ES) |

**Training order:** V → M → C (sequential, not simultaneous)

---

## When to Use Model-Based RL

**Use model-based RL when:**

- Real environment interactions are expensive or slow (robotics, wet-lab experiments, dangerous environments)
- Sample budget is limited (fewer than ~100,000 real steps)
- Environment dynamics are learnable (smooth, low-dimensional structure)
- Planning at decision time is needed (safety constraints, multi-step lookahead)
- A good simulator already exists or can be built (use MCTS / MuZero)

**Prefer model-free when:**

- Fast simulation is available (Atari, MuJoCo with GPU sim)
- Dynamics are too complex to model accurately (contact-rich physics, multi-agent adversarial)
- Engineering simplicity is a priority (fewer components, easier to debug)
- Large-scale distributed training is feasible (PPO with many workers)

**Signs that your model-based approach is working:**
- Model loss (MSE on held-out transitions) is decreasing and plateauing
- Planning steps improve performance compared to pure model-free baseline
- Model-generated rollouts are visually plausible (if observation-space model)

**Signs that your model-based approach is failing:**
- Policy performance worsens as planning steps $n$ increase (model exploitation)
- Model loss keeps decreasing but policy performance does not improve (model is accurate but irrelevant)
- High variance in policy returns (model errors amplifying across rollout horizon)

---

## Quick Reference: Algorithm Properties

| Algorithm | Year | Model Type | Planning | Key Hyperparameters |
|-----------|------|-----------|---------|---------------------|
| Dyna-Q | 1991 | Tabular $(s,a) \to (r, s')$ | Background, $n$ steps/real-step | $n$, $\alpha$, $\gamma$, $\epsilon$ |
| Dyna-Q+ | 1991 | Tabular + staleness | Background, $n$ steps | $n$, $\kappa$ |
| MCTS | 2006 | Simulator (any) | Decision-time, $K$ simulations | $K$, $c$, rollout depth $H$ |
| AlphaZero | 2017 | Perfect simulator + NN | Decision-time MCTS | $c$, $K$, self-play |
| World Models | 2018 | VAE + MDN-RNN | Dream training | $z$-dim, RNN hidden, $\beta$ |
| MuZero | 2020 | Latent dynamics NN | Decision-time MCTS | $K$, unroll depth, $n$-step |
| MBPO | 2019 | Ensemble MLPs | Background, short rollouts | $H$, ensemble size $M$ |
| DreamerV3 | 2023 | RSSM (latent) | Background dream | Fixed across all domains |

---

## Notation Summary (Module 8)

| Symbol | Meaning |
|--------|---------|
| $p(s' \mid s, a)$ | True transition probability |
| $\hat{p}_\theta(s' \mid s, a)$ | Learned transition model |
| $\hat{r}_\phi(s, a)$ | Learned reward model |
| $n$ | Number of Dyna-Q planning steps per real step |
| $K$ | Number of MCTS simulations |
| $H$ | MCTS rollout horizon or model rollout length |
| $c$ | UCT exploration coefficient |
| $N(s)$ | MCTS visit count for state $s$ |
| $N(s, a)$ | MCTS visit count for action $a$ from state $s$ |
| $\bar{Q}(s, a)$ | MCTS empirical mean return from $(s, a)$ |
| $z_t$ | Latent state (World Models VAE encoding) |
| $h_t$ | RNN hidden state (World Models memory) |
| $h_k$ | MuZero latent hidden state at step $k$ |
| $r_k, h_k = g(h_{k-1}, a_k)$ | MuZero dynamics function |
| $p_k, v_k = f(h_k)$ | MuZero prediction function |
| $h_0 = h(o_1, \ldots, o_t)$ | MuZero representation function |
| $\tau(s, a)$ | Steps since $(s,a)$ was last tried (Dyna-Q+) |
| $\kappa$ | Dyna-Q+ exploration bonus coefficient |

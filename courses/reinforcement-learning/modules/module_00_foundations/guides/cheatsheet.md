# Reinforcement Learning Foundations — Cheatsheet

> **Reading time:** ~5 min | **Module:** 0 — Foundations | **Prerequisites:** Probability, linear algebra

## Symbol Table

| Symbol | Name | Definition |
|--------|------|------------|
| $s \in \mathcal{S}$ | State | A representation of the environment at one point in time |
| $a \in \mathcal{A}$ | Action | A choice made by the agent |
| $r \in \mathbb{R}$ | Reward | Scalar feedback received after each action |
| $\pi(a \mid s)$ | Policy | Probability of selecting action $a$ in state $s$ |
| $V^\pi(s)$ | State-value function | Expected return from state $s$ under policy $\pi$ |
| $Q^\pi(s, a)$ | Action-value function | Expected return from $(s, a)$, then following $\pi$ |
| $\gamma \in [0, 1)$ | Discount factor | Down-weights future rewards relative to immediate ones |
| $\alpha \in (0, 1]$ | Learning rate | Step size for value function updates (TD, Q-learning) |
| $\epsilon \in [0, 1]$ | Exploration rate | Probability of selecting a random action ($\epsilon$-greedy) |
| $G_t$ | Return | Discounted sum of future rewards from time $t$ |
| $p(s', r \mid s, a)$ | Transition dynamics | Joint probability of next state and reward |
| $\tau$ | Trajectory | Sequence $(S_0, A_0, R_1, S_1, A_1, R_2, \ldots)$ |
| $T$ | Episode length | Time step of terminal state in episodic tasks |
| $\delta_t$ | TD error | $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ |

---

## MDP Tuple

$$\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ p,\ R,\ \gamma)$$

<div class="callout-insight">
<strong>Insight:</strong> $$\mathcal{M} = (\mathcal{S},\ \mathcal{A},\ p,\ R,\ \gamma)$$

| Component | Description | Constraint |
|-----------|-------------|------------|
| $\mathcal{S}$ | State space | Finite or continuous |...
</div>


| Component | Description | Constraint |
|-----------|-------------|------------|
| $\mathcal{S}$ | State space | Finite or continuous |
| $\mathcal{A}$ | Action space | Finite or continuous |
| $p(s', r \mid s, a)$ | Transition dynamics | $\sum_{s'} \sum_r p(s', r \mid s, a) = 1$ |
| $R$ | Reward (implied by $p$) | — |
| $\gamma$ | Discount factor | $0 \leq \gamma < 1$ (or $\gamma = 1$ episodic only) |

**Markov property:**
$$p(s_{t+1}, r_{t+1} \mid s_t, a_t) = p(s_{t+1}, r_{t+1} \mid s_1, a_1, \ldots, s_t, a_t)$$

---

## Returns

**General discounted return:**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

<div class="callout-key">
<strong>Key Point:</strong> **General discounted return:**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Episodic (finite horizon $T$):**
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

**Recursive decomposition (foundati...
</div>


**Episodic (finite horizon $T$):**
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

**Recursive decomposition (foundation of all Bellman equations):**
$$G_t = R_{t+1} + \gamma G_{t+1}$$

---

## All Four Bellman Equations

### Bellman Expectation — State-Value

<div class="callout-info">
<strong>Info:</strong> ### Bellman Expectation — State-Value

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]$$

### Bellman Expectation — Action-Value

$$Q^\pi(s, a) = \...
</div>


$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]$$

### Bellman Expectation — Action-Value

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s', a')\right]$$

### Bellman Optimality — State-Value

$$V^*(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^*(s')\right]$$

### Bellman Optimality — Action-Value

$$Q^*(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma \max_{a'} Q^*(s', a')\right]$$

### Key Relationships

$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$$

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]$$

$$V^*(s) = \max_a Q^*(s, a)$$

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

---

## Key Distinctions

### Episodic vs Continuing Tasks

<div class="callout-key">
<strong>Key Point:</strong> ### Episodic vs Continuing Tasks

| | Episodic | Continuing |
|--|---------|-----------|
| Terminal state?
</div>


| | Episodic | Continuing |
|--|---------|-----------|
| Terminal state? | Yes ($\mathcal{S}^+$) | No |
| Return | Finite sum (may use $\gamma = 1$) | Must use $\gamma < 1$ |
| Example | Game, trading day | Process control, portfolio |
| Episode resets? | Yes | No |

### On-Policy vs Off-Policy

| | On-Policy | Off-Policy |
|--|-----------|-----------|
| Evaluates | Behavior policy $\pi$ | Different target policy $\pi^*$ |
| Bellman target | $r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a')$ | $r + \gamma \max_{a'} Q(s', a')$ |
| Algorithms | SARSA, Monte Carlo, A3C | Q-learning, DQN, SAC |
| Experience replay? | Requires care | Natural fit |

### Model-Based vs Model-Free

| | Model-Based | Model-Free |
|--|-------------|-----------|
| Requires $p(s', r \mid s, a)$? | Yes | No |
| Plans ahead? | Yes | No — learns from experience |
| Sample efficiency | Higher | Lower (but no model error) |
| Algorithms | Value iteration, policy iteration, Dyna | Q-learning, DQN, PPO, SAC |

---

## $\epsilon$-Greedy Policy

$$\pi_\epsilon(a \mid s) = \begin{cases} 1 - \epsilon + \dfrac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \dfrac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

With probability $1 - \epsilon$: exploit (take greedy action).
With probability $\epsilon$: explore (take random action).

---

## Temporal-Difference Update (Q-learning)

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \underbrace{\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\right]}_{\delta_t \text{ — TD error}}$$

**SARSA (on-policy variant):**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$$

---

## Discount Factor Reference

| $\gamma$ | Effective horizon (steps) | Interpretation |
|----------|--------------------------|----------------|
| 0.50 | $\approx 1$ | Nearly myopic |
| 0.90 | $\approx 10$ | Short-horizon planning |
| 0.95 | $\approx 20$ | Medium-horizon |
| 0.99 | $\approx 100$ | Long-horizon planning |
| 0.999 | $\approx 1000$ | Very far-sighted |

Effective horizon $\approx \dfrac{1}{1-\gamma}$

---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Q-Learning: Off-Policy TD Control

## Module 3: Temporal Difference Learning
### Reinforcement Learning

<!-- Speaker notes: Q-learning is arguably the most important algorithm in the entire course — it is the direct ancestor of DQN, which sparked the deep RL revolution. The key concept: Q-learning is off-policy because its update target uses the greedy max action, not the action actually taken. This allows Q-learning to learn the optimal policy regardless of how the agent explores. Watkins and Dayan (1992) proved this rigorously. -->

---

## One Change from SARSA

<div class="columns">
<div>

**SARSA (on-policy)**
$$Q(S,A) \leftarrow Q(S,A)$$
$$+ \alpha[R + \gamma \underbrace{Q(S', A')}_{\text{actual next action}} - Q(S,A)]$$

</div>
<div>

**Q-learning (off-policy)**
$$Q(S,A) \leftarrow Q(S,A)$$
$$+ \alpha[R + \gamma \underbrace{\max_a Q(S', a)}_{\text{greedy max}} - Q(S,A)]$$

</div>
</div>

> $A'$ vs $\max_a$ — one word, two completely different algorithms.

Q-learning's target does not depend on the behavior policy. It always points toward the greedy optimum.

<!-- Speaker notes: Have students look at both equations simultaneously. The only difference is in the bootstrap term. SARSA uses Q(S', A') where A' is actually sampled from the behavior policy. Q-learning uses max_a Q(S', a) — the best action according to Q, whether or not it was taken. This makes Q-learning off-policy: the target policy (greedy) differs from the behavior policy (epsilon-greedy). -->

---

## Off-Policy Learning

**Two distinct policies coexist in Q-learning:**

```
Behavior policy b:  ε-greedy over Q   →  generates transitions (S, A, R, S')
Target policy  π:  greedy over Q      →  defines the learning target
```

Off-policy learning separates *how we explore* from *what we learn*.

**Consequence:** Q-learning directly approximates $Q^*$ — the optimal action-value function — even while following a random or suboptimal behavior policy.

> This is the key feature that enables experience replay, human-guided exploration, and transfer learning.

<!-- Speaker notes: The separation of behavior and target policies is one of the most powerful ideas in RL. Because Q-learning is off-policy, you can: (1) use a separate random policy to collect data and still learn optimally, (2) learn from old data stored in a replay buffer (DQN), (3) learn from human demonstrations. SARSA cannot do any of these things without additional care, because it must follow the behavior policy it evaluates. -->

---

## What Q-learning Learns

Q-learning converges to $Q^*$, which satisfies the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\bigl[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t=s, A_t=a\bigr]$$

The optimal greedy policy is then recovered for free:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

> Once you have $Q^*$, you have the optimal policy. No additional policy extraction step needed.

<!-- Speaker notes: This is why Q-learning is so popular: it directly learns Q*, which encodes the optimal policy. Compare to prediction methods (TD(0), MC) which only evaluate a fixed policy — you still need a separate policy improvement step. Q-learning collapses evaluation and improvement into a single update. The Bellman optimality equation is the fixed point that Q-learning converges to. -->

---

## Convergence: Watkins & Dayan (1992)

**Q-learning converges to $Q^*$ with probability 1 given:**

1. Every $(s, a)$ pair visited infinitely often
2. Step sizes satisfy Robbins-Monro: $\displaystyle\sum_t \alpha_t = \infty$, $\displaystyle\sum_t \alpha_t^2 < \infty$
3. Rewards are bounded: $|R| \leq R_{\max}$

**Proof key insight:** Q-learning is a stochastic approximation of the operator $(\mathcal{T}^* Q)(s,a) = \mathbb{E}[R + \gamma \max_{a'} Q(S', a')]$.

Since $\mathcal{T}^*$ is a $\gamma$-contraction in $\ell^\infty$, it has a unique fixed point $Q^*$, and the Robbins-Monro conditions ensure convergence of the stochastic iterates.

<!-- Speaker notes: The convergence proof is a landmark result. Students do not need to reproduce it, but should understand the structure: Q-learning is an instance of stochastic approximation applied to the Bellman optimality operator. The contraction property of T* is what guarantees a unique fixed point. The Robbins-Monro conditions handle the stochastic noise. Together they guarantee that Q converges to Q* as the number of visits grows. -->

---

## Maximization Bias

**The problem:** Q-learning always takes $\max_a Q(S', a)$ of noisy estimates.

```
True Q*(s, a) = 0 for all a
Estimated Q(s, a_1) = +0.3  ← noise
Estimated Q(s, a_2) = -0.1  ← noise
Estimated Q(s, a_3) = +0.5  ← noise

max_a Q(s, a) = +0.5   ← biased upward from 0!
```

**The maximum of noisy zero-mean estimates is always positive.**

This systematic overestimation is **maximization bias** — Q-learning is overoptimistic during learning.

<!-- Speaker notes: Maximization bias is subtle but real. If you have 10 actions each with true value 0, their Q estimates will vary randomly around 0. The maximum of these estimates is always positive. So Q-learning consistently overestimates the value of the next state. This propagates backwards through the Bellman backup and can cause Q-learning to overvalue certain actions persistently. Double Q-learning fixes this. -->

---

## Double Q-Learning: The Fix

Use two independent Q tables. Decouple action selection from action evaluation.

**Update (with probability 0.5, update $Q_1$):**

$$Q_1(S,A) \leftarrow Q_1(S,A) + \alpha \Bigl[R + \gamma Q_2\bigl(S', \underbrace{\arg\max_a Q_1(S',a)}_{\text{selected by }Q_1}\bigr) - Q_1(S,A)\Bigr]$$

| Step | Uses |
|------|------|
| Select next action | $Q_1$: $a^* = \arg\max_a Q_1(s', a)$ |
| Evaluate next action | $Q_2$: $Q_2(s', a^*)$ |

> Because $Q_1$ and $Q_2$ are independent, the selection-evaluation independence eliminates the upward bias.

<!-- Speaker notes: The key insight: the bias comes from using the same estimate to both select and evaluate the best action. If we use Q1 to pick the best action but Q2 to evaluate it, the estimates are independent — so the maximum selection of Q1 does not inflate the evaluation by Q2. Empirically, Double Q-learning produces much more accurate value estimates in early training. DQN uses a variant of this idea (target networks). -->

---

## Code: Q-Learning Core

```python
import numpy as np

def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(num_episodes):
        state, _ = env.reset()

        while True:
            # Behavior: epsilon-greedy (explores)
            action = np.argmax(Q[state]) if np.random.random() > epsilon \
                     else env.action_space.sample()

            next_state, reward, terminated, truncated, _ = env.step(action)

            # Target: greedy max (off-policy)
            next_max = 0.0 if terminated else np.max(Q[next_state])
            td_error = reward + gamma * next_max - Q[state, action]
            Q[state, action] += alpha * td_error

            if terminated or truncated:
                break
            state = next_state
    return Q
```

> Unlike SARSA: no carry-forward of `next_action`. Q-learning uses `np.max` — the action is never actually needed.

<!-- Speaker notes: Highlight the critical difference from SARSA code: there is no next_action variable. Q-learning never needs to know which action was taken next — it just needs the value of the best possible action. This is what makes it off-policy: the actual next action could be anything. The bootstrap uses max_a Q[next_state], which does not depend on what the agent does next. -->

---

## Code: Double Q-Learning

```python
def double_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q1, Q2 = np.zeros((n_s, n_a)), np.zeros((n_s, n_a))

    for _ in range(num_episodes):
        state, _ = env.reset()
        while True:
            # Behavior: epsilon-greedy over combined Q1+Q2
            combined_action = np.argmax(Q1[state] + Q2[state]) \
                              if np.random.random() > epsilon \
                              else env.action_space.sample()
            ns, r, term, trunc, _ = env.step(combined_action)

            if np.random.random() < 0.5:    # Update Q1
                a_star = np.argmax(Q1[ns])  # Select with Q1
                val = 0.0 if term else Q2[ns, a_star]  # Evaluate with Q2
                Q1[state, combined_action] += alpha * (r + gamma * val - Q1[state, combined_action])
            else:                            # Update Q2
                a_star = np.argmax(Q2[ns])  # Select with Q2
                val = 0.0 if term else Q1[ns, a_star]  # Evaluate with Q1
                Q2[state, combined_action] += alpha * (r + gamma * val - Q2[state, combined_action])

            if term or trunc: break
            state = ns
    return Q1, Q2
```

<!-- Speaker notes: Walk through the double Q-learning code. The key pattern: randomly decide which table to update, then use that table for action selection and the OTHER table for evaluation. The 50/50 random split ensures both tables are updated equally. The final estimate uses (Q1 + Q2) / 2. In DQN, this becomes "online network" (Q1) and "target network" (Q2), with the target network updated periodically rather than randomly. -->

---

## Cliff Walking: Algorithm Comparison

<div class="columns">
<div>

**SARSA**
- Path: upper route (safe)
- Accounts for exploration accidents
- Higher training reward
- Suboptimal deployment policy

</div>
<div>

**Q-learning**
- Path: cliff edge (optimal)
- Ignores exploration noise
- Lower training reward (cliff falls)
- Optimal deployment policy

</div>
</div>

```
Training returns (500 episodes, ε=0.1):
SARSA:      avg ≈ -25 per episode
Q-learning: avg ≈ -55 per episode

Greedy deployment returns:
SARSA:      avg ≈ -17 (upper path)
Q-learning: avg ≈ -13 (optimal path)
```

<!-- Speaker notes: The numbers here illustrate the core tradeoff. During training, SARSA does better because it avoids cliff falls. After training, Q-learning does better because it learned the optimal path. The "right" choice depends entirely on whether training environment cost matters. In real robotics: SARSA (physical damage from cliff falls). In simulation: Q-learning (fast convergence to optimal). -->

---

<!-- _class: lead -->

# Common Pitfalls

<!-- Speaker notes: The Q-learning pitfalls mostly concern understanding what "off-policy" means in practice. The most important one: Q-learning does not automatically become better just by using more exploration — you need the right exploration-exploitation balance. -->

---

## Pitfall 1: On-Policy Confusion

```python
# WRONG: accidentally wrote SARSA while calling it Q-learning
next_action = epsilon_greedy(Q, next_state, epsilon)
td_target = reward + gamma * Q[next_state, next_action]  # ← SARSA!

# CORRECT Q-learning:
td_target = reward + gamma * np.max(Q[next_state])       # ← off-policy max
```

**Test:** On cliff walking, does your agent learn the cliff-edge path (Q-learning) or the upper path (SARSA)? If you expect Q-learning but see the upper path, check your target.

<!-- Speaker notes: The single fastest way to debug this: run cliff walking, extract the greedy policy, and check whether it uses the cliff-edge path or the upper path. If you see the upper path, you wrote SARSA. If you see the cliff edge, you wrote Q-learning. This behavioral test is more reliable than code inspection alone. -->

---

## Pitfall 2: Forgetting Maximization Bias

| Environment | Action space | Bias severity | Recommendation |
|-------------|-------------|---------------|----------------|
| FrozenLake (4 actions) | Small | Mild | Standard Q-learning fine |
| CliffWalking (4 actions) | Small | Mild | Standard Q-learning fine |
| Atari (18 actions) | Large | Severe | Use Double Q-learning |
| Continuous (many bins) | Very large | Very severe | Use Double Q-learning |

> As the number of actions grows, maximization bias worsens. Double Q-learning has negligible extra cost — use it by default when action spaces are large.

<!-- Speaker notes: Maximization bias is proportional to the number of actions and the variance of rewards. With 4 actions, it is usually tolerable. With 18 (Atari) or more, it can cause Q-values to be off by 30-50% in early training. Double DQN (the deep RL version of Double Q-learning) became standard precisely because vanilla DQN's overestimation was clearly measurable in Atari benchmarks. -->

---

## Pitfall 3: No Exploration → No Convergence

**Convergence requires every $(s,a)$ visited infinitely often.**

| $\varepsilon$ | Behavior |
|-----------|----------|
| 0.0 | Greedy from start; most actions never tried |
| 0.01 | Very little exploration; convergence very slow |
| 0.1 | Common practical choice |
| 0.5 | Lots of exploration; Q values remain noisy |

> A completely greedy Q-learning agent ($\varepsilon = 0$) cannot converge to $Q^*$ because it never discovers suboptimal paths that might lead to better long-term outcomes.

<!-- Speaker notes: This pitfall is counterintuitive: Q-learning with no exploration is greedy but not optimal. Greedy behavior from random initialization is like reading only the first chapter of a book and concluding you know the whole story. Exploration is not optional — it is the data collection mechanism. Without it, many state-action pairs are never visited and their Q-values remain at the initial (wrong) values. -->

---

## Summary

<div class="columns">
<div>

### Q-Learning Facts
- Off-policy TD control
- Learns $Q^*$ directly
- Behavior policy can be anything
- Uses $\max_a Q(S',a)$ as target
- Watkins & Dayan (1992) convergence proof

</div>
<div>

### Update Rule
$$Q(S,A) \leftarrow Q(S,A)$$
$$+ \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]$$

Double Q-learning (for max bias):
$$Q_1(S,A) \leftarrow Q_1(S,A)$$
$$+ \alpha[R + \gamma Q_2(S', \arg\max_a Q_1(S',a)) - Q_1(S,A)]$$

</div>
</div>

**Next:** TD(λ) unifies TD(0) and Monte Carlo via n-step returns and eligibility traces.

<!-- Speaker notes: Close by previewing TD(lambda). Q-learning and SARSA use one-step backups. What if we used 2 steps? 10 steps? All steps? TD(lambda) provides a principled way to interpolate between these extremes. The parameter lambda controls the depth of the backup — lambda=0 gives TD(0), lambda=1 gives Monte Carlo. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- SARSA on-policy control (Guide 02)
- TD(0) prediction (Guide 01)
- Bellman optimality equation (Module 0)
- Value iteration (Module 01)

</div>
<div>

### Leads To
- Deep Q-Network — DQN (Module 05)
- Double DQN, Dueling DQN
- Prioritized Experience Replay
- Q-learning with function approximation (Module 04)
- Distributional RL

</div>
</div>

**Historical note:** DQN (Mnih et al., 2015) = Q-learning + neural network + experience replay + target network. The 2015 Nature paper launched the deep RL era, achieving human-level performance on 49 Atari games.

<!-- Speaker notes: Q-learning's legacy is enormous. DQN used it unchanged as the core update rule — the only additions were engineering to make it stable with neural networks (experience replay to break correlations, target network to stabilize the bootstrap target). When students later study DQN, they will recognize the Q-learning update immediately. This is a satisfying moment of connection — the simple tabular algorithm from this guide becomes the backbone of a system that plays Atari at human level. -->


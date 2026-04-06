# Monte Carlo Control

> **Reading time:** ~12 min | **Module:** 2 — Monte Carlo Methods | **Prerequisites:** Module 1

## In Brief

Monte Carlo control finds the optimal policy $\pi^*$ without a model by alternating between MC policy evaluation and greedy policy improvement. Because improvement requires comparing actions, we estimate $Q(s,a)$ (action-value) rather than $V(s)$ (state-value).

<div class="callout-key">

<strong>Key Concept:</strong> Monte Carlo control finds the optimal policy $\pi^*$ without a model by alternating between MC policy evaluation and greedy policy improvement. Because improvement requires comparing actions, we estimate $Q(s,a)$ (action-value) rather than $V(s)$ (state-value).

</div>


## Key Insight

DP control used $V(s)$ but needed the model to do $\arg\max_a \sum_{s'} p(s'|s,a)[r + \gamma V(s')]$. Without a model, we cannot do that argmax. Estimating $Q(s,a)$ directly makes the greedy improvement step model-free: $\pi(s) = \arg\max_a Q(s,a)$.

---


<div class="callout-key">

<strong>Key Point:</strong> DP control used $V(s)$ but needed the model to do $\arg\max_a \sum_{s'} p(s'|s,a)[r + \gamma V(s')]$.

</div>

## Why $Q(s,a)$, Not $V(s)$, for Model-Free Control

In Module 1, policy improvement used:

<div class="callout-insight">

<strong>Insight:</strong> In Module 1, policy improvement used:

$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]$$

This requires the transition model $p(s', r \mid s, a)$.

</div>


$$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]$$

This requires the transition model $p(s', r \mid s, a)$.

With $Q^\pi(s, a)$ the improvement step becomes:

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

No model needed. This is the central reason MC control targets $Q$.

**Formal definition:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]$$

The expected return when taking action $a$ in state $s$, then following $\pi$ thereafter.

---

## Generalized Policy Iteration (GPI) with MC

MC control is an instance of Generalized Policy Iteration (GPI):

<div class="callout-info">

<strong>Info:</strong> MC control is an instance of Generalized Policy Iteration (GPI):


Each evaluation step uses MC episodes to update $Q$.

</div>


```
        Evaluate
Q ←────────────────── π
 │                     ↑
 │    Improve           │
 └──────────────────────┘
     π(s) = argmax_a Q(s,a)
```

Each evaluation step uses MC episodes to update $Q$. Each improvement step updates $\pi$ greedily with respect to the current $Q$. The two steps alternate, driving $Q$ toward $Q^*$ and $\pi$ toward $\pi^*$.

**The challenge:** If we evaluate $Q^\pi$ fully before improving, we waste episodes. In practice, we improve after every episode (or even every visit).

---

## Exploring Starts

For MC to estimate $Q(s,a)$ for every state-action pair, every pair must be visited. If the policy is deterministic, many $(s, a)$ pairs are never visited and $Q(s, a)$ remains unknown for unchosen actions.

<div class="callout-warning">

<strong>Warning:</strong> For MC to estimate $Q(s,a)$ for every state-action pair, every pair must be visited.

</div>


**Exploring Starts assumption (ES):** Each episode begins with a randomly chosen starting state-action pair $(S_0, A_0)$, with all pairs having non-zero probability of being selected as the start.

This guarantees all $(s, a)$ pairs are visited infinitely often in the limit.

**When ES is feasible:**
- Simulated environments where start state can be set arbitrarily
- Games where any board position can be initialized
- Training scenarios with full environment control

**When ES is infeasible:**
- Real robots (cannot teleport to arbitrary states)
- Online learning from a single trajectory
- Environments with fixed initial states (Atari, for example)

When ES is not feasible, use $\epsilon$-soft policies instead.

---

## Algorithm: MC ES (Monte Carlo with Exploring Starts)

**Input:** Discount $\gamma$

**Output:** Optimal policy $\pi^*$, optimal $Q^*$

```
Initialize:
  Q(s, a) = 0 for all s ∈ S, a ∈ A(s)
  π(s) = arbitrary for all s ∈ S
  Returns(s, a) = empty list for all s, a

Loop for each episode:
  1. Choose S_0, A_0 randomly such that all pairs
     have non-zero probability (exploring start)

  2. Generate episode from (S_0, A_0) following π:
     S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T

  3. G = 0
     For t = T-1, T-2, ..., 0:
       G = γ · G + R_{t+1}
       If (S_t, A_t) not in {(S_0, A_0), ..., (S_{t-1}, A_{t-1})}:
         Append G to Returns(S_t, A_t)
         Q(S_t, A_t) = average(Returns(S_t, A_t))
         π(S_t) = argmax_a Q(S_t, a)  ← greedy improvement
```

Note: $\pi$ is updated immediately at each $(s, a)$ visit. This is MC ES with in-episode improvement.

---

## On-Policy MC Control: $\epsilon$-Soft Policies

When exploring starts cannot be guaranteed, the policy itself must ensure exploration. An **$\epsilon$-soft policy** assigns non-zero probability to every action:

$$\pi(a \mid s) \geq \frac{\varepsilon}{|\mathcal{A}(s)|} \quad \forall a \in \mathcal{A}(s), \forall s \in \mathcal{S}$$

The simplest $\epsilon$-soft policy is **$\epsilon$-greedy**:

$$\pi(a \mid s) = \begin{cases}
1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\varepsilon}{|\mathcal{A}(s)|} & \text{otherwise}
\end{cases}$$

With probability $1 - \varepsilon$: exploit (take the greedy action).
With probability $\varepsilon$: explore (choose uniformly at random).

**On-policy** means we both generate episodes from $\pi$ and learn about $\pi$. The policy we evaluate is the same policy we improve.

---

## Algorithm: On-Policy First-Visit MC Control

**Input:** $\varepsilon \in (0, 1)$, discount $\gamma$, number of episodes $N$

**Output:** Approximately optimal $\epsilon$-soft policy $\pi$

```
Initialize:
  Q(s, a) = 0 for all s ∈ S, a ∈ A(s)
  N(s, a) = 0 for all s, a
  π(a|s) = 1/|A(s)| for all s (uniform, which is ε-soft)

For each episode i = 1, ..., N:
  1. Generate episode using current π:
     S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T

  2. G = 0
     For t = T-1, T-2, ..., 0:
       G = γ · G + R_{t+1}
       If (S_t, A_t) not in {(S_j, A_j) : j < t}:  # first-visit
         N(S_t, A_t) += 1
         Q(S_t, A_t) += (G - Q(S_t, A_t)) / N(S_t, A_t)

         # ε-greedy improvement
         A* = argmax_a Q(S_t, a)
         For all a ∈ A(S_t):
           if a == A*:
             π(a|S_t) = 1 - ε + ε/|A(S_t)|
           else:
             π(a|S_t) = ε/|A(S_t)|
```

---

## GLIE: Greedy in the Limit with Infinite Exploration

An on-policy MC control algorithm converges to $Q^*$ if the policy sequence satisfies the **GLIE condition** (Sutton & Barto, Ch. 5.4):

1. **Infinite exploration:** All state-action pairs are visited infinitely often:
   $$\lim_{k \to \infty} N_k(s, a) = \infty \quad \forall s, a$$

2. **Greedy in the limit:** The policy converges to greedy with respect to $Q$:
   $$\lim_{k \to \infty} \pi_k(a \mid s) = \mathbf{1}\left[a = \arg\max_{a'} Q_k(s, a')\right]$$

A standard schedule that satisfies GLIE: $\varepsilon_k = \frac{1}{k}$ (decays as $1/\text{episode number}$). This satisfies:
- Condition 1 because every action gets probability $\varepsilon_k / |\mathcal{A}|$ at every step
- Condition 2 because $\varepsilon_k \to 0$, making the policy increasingly greedy

**Practical note:** $\varepsilon = 1/k$ converges very slowly. In practice, a linear or exponential decay schedule is more effective, though it technically violates GLIE's infinite-exploration requirement.

---

## Python Implementation


The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from collections import defaultdict


def make_epsilon_greedy_policy(Q, epsilon, n_actions):
    """
    Create an epsilon-greedy policy from a Q-table.

    Returns a function: state -> probability array over actions.
    """
    def policy(state):
        probs = np.ones(n_actions) * epsilon / n_actions
        best_action = np.argmax(Q[state])
        probs[best_action] += 1.0 - epsilon
        return probs
    return policy


def mc_control_epsilon_greedy(env, n_episodes=50_000, gamma=1.0,
                               epsilon=0.1, epsilon_decay=True):
    """
    On-policy first-visit MC control with epsilon-greedy exploration.

    Args:
        env:           Gymnasium episodic environment
        n_episodes:    Number of training episodes
        gamma:         Discount factor
        epsilon:       Initial exploration probability
        epsilon_decay: If True, decay epsilon as 1/episode_number (GLIE)

    Returns:
        Q:      dict mapping (state, action) -> estimated Q value
        policy: callable state -> action (greedy w.r.t. final Q)
    """
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    N = defaultdict(lambda: np.zeros(n_actions))  # visit counts

    for episode_num in range(1, n_episodes + 1):
        eps = 1.0 / episode_num if epsilon_decay else epsilon

        # Generate episode using current epsilon-greedy policy
        episode = []
        state, _ = env.reset()
        done = False

        while not done:
            probs = np.ones(n_actions) * eps / n_actions
            probs[np.argmax(Q[state])] += 1.0 - eps
            action = np.random.choice(n_actions, p=probs)

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            done = terminated or truncated
            state = next_state

        # Backward pass: compute returns and update Q
        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = gamma * G + reward

            if (state, action) not in visited:        # first-visit
                visited.add((state, action))
                N[state][action] += 1
                # Incremental mean update
                Q[state][action] += (G - Q[state][action]) / N[state][action]

    # Extract greedy policy from final Q
    def greedy_policy(state):
        return int(np.argmax(Q[state]))

    return Q, greedy_policy
```

</div>
</div>

---

## Diagram: MC Control Information Flow

```
                  ┌─────────────────────────────────────┐
                  │           ENVIRONMENT               │
                  └───────────────┬─────────────────────┘
                                  │ episode: (S,A,R) tuples
                                  ▼
                  ┌─────────────────────────────────────┐
                  │        BACKWARD PASS                │
                  │   G_t = R_{t+1} + γ G_{t+1}        │
                  └───────────────┬─────────────────────┘
                                  │ (S_t, A_t, G_t) for first visits
                                  ▼
                  ┌─────────────────────────────────────┐
                  │         Q-TABLE UPDATE              │
                  │   Q(S_t,A_t) += (G_t - Q(S_t,A_t)) │
                  │               ──────────────────    │
                  │                   N(S_t,A_t)        │
                  └───────────────┬─────────────────────┘
                                  │ updated Q
                                  ▼
                  ┌─────────────────────────────────────┐
                  │       ε-GREEDY IMPROVEMENT          │
                  │   π(s) ← ε-greedy w.r.t. Q(s,·)   │
                  └───────────────┬─────────────────────┘
                                  │ updated policy
                                  └──────────────────────→ (next episode)
```

---

## Convergence: Policy Improvement Theorem for $\epsilon$-Soft Policies

For any $\epsilon$-soft policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $Q^\pi$ satisfies:

$$Q^{\pi'}(s, a) \geq Q^\pi(s, a) \quad \forall s, a$$

**Proof sketch:** The $\epsilon$-greedy policy allocates more probability to higher-$Q$ actions. Computing $V^{\pi'}(s) - V^\pi(s)$ and expanding via the Bellman equations shows the difference is non-negative (Sutton & Barto Ch. 5.4, Theorem 5.2).

**Caveat:** The optimal policy for $\epsilon$-soft policies is the best $\epsilon$-soft policy, which is not $\pi^*$ (the unconstrained optimum). As $\varepsilon \to 0$, the optimal $\epsilon$-soft policy approaches $\pi^*$.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Estimating $V$ instead of $Q$**
MC control requires $Q(s,a)$, not $V(s)$. A classic mistake is attempting to improve the policy from $V(s)$ estimates without a model. Without $p(s'|s,a)$, the greedy improvement step $\arg\max_a \sum_{s'} p(s'|s,a)[r + \gamma V(s')]$ is undefined. Always collect action-value returns.

<div class="callout-warning">

<strong>Warning:</strong> **Estimating $V$ instead of $Q$**
MC control requires $Q(s,a)$, not $V(s)$.

</div>

**Never exploring away from the greedy action**
With a fixed deterministic policy, non-greedy actions accumulate no returns. $Q(s,a)$ for non-greedy $a$ stays at its initial value forever, and the argmax over $Q$ never changes. Solution: $\epsilon$-greedy or exploring starts.

**$\varepsilon$ too large or too small**
Large $\varepsilon$: lots of exploration, poor exploitation, policy never converges to greedy. Small $\varepsilon$: fast convergence to suboptimal local policy, insufficient exploration. Decaying $\varepsilon$ (GLIE schedule) balances both over time.

**Episode length and discount mismatch**
For tasks with $\gamma = 1.0$ (undiscounted), very long episodes cause returns at early timesteps to be large and noisy. Consider $\gamma < 1$ to bound the return magnitude and reduce variance.

**Stale Q-values after policy change**
After a policy improvement step, the Q-values for the old policy are no longer correct for the new policy. In practice this is fine — GPI converges despite this imprecision — but be aware that estimates lag behind policy changes. More improvement steps = more staleness.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** MC prediction (Guide 01), GPI framework (Module 1 — DP control)
- **Leads to:** Importance sampling (Guide 03) for off-policy Q-estimation; TD control (SARSA, Q-learning) in Module 3
- **Related to:** Multi-armed bandits ($\varepsilon$-greedy exploration), function approximation (Module 4 extends tabular Q to neural Q-networks)


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 5.3–5.4 — primary reference
- Singh, Jaakkola & Jordan (2000), "Convergence results for single-step on-policy RL algorithms" — formal GLIE convergence proof
- Mnih et al. (2015), "Human-level control through deep reinforcement learning" — DQN applies MC-style Q-estimation with neural function approximation


---

## Cross-References

<a class="link-card" href="./02_monte_carlo_control_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_mc_prediction.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>

# Q-Learning: Off-Policy TD Control

## In Brief

Q-learning is an off-policy TD control algorithm that directly learns the optimal action-value function $Q^*$ regardless of the policy used to generate experience. It is one of the most important RL algorithms ever developed: with Watkins & Dayan (1992), it provided the first rigorous proof that a simple, practical algorithm can converge to optimal behavior.

## Key Insight

Q-learning's update target uses $\max_a Q(S_{t+1}, a)$ — the *best possible* next action value — rather than the value of the action actually taken. This one change decouples the learning target from the behavior policy, making Q-learning off-policy and allowing it to converge to $Q^*$ even while the agent explores randomly.

---

## Formal Definition

### Q-Learning Update Rule

After observing the transition $(S_t, A_t, R_{t+1}, S_{t+1})$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]$$

| Symbol | Meaning |
|--------|---------|
| $Q(S_t, A_t)$ | Current estimate of optimal action value |
| $\alpha$ | Step size |
| $R_{t+1}$ | Reward received |
| $\gamma$ | Discount factor |
| $\max_a Q(S_{t+1}, a)$ | Best known action value in next state (greedy target) |

### TD Error for Q-Learning

$$\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$$

The update: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \, \delta_t$

---

## Off-Policy: Separating Behavior from Target

**Off-policy learning** uses two distinct policies:

| Policy | Role | In Q-learning |
|--------|------|---------------|
| **Behavior policy** $b$ | Generates experience (actions taken) | $\varepsilon$-greedy over $Q$ |
| **Target policy** $\pi$ | Policy being evaluated/improved | Greedy $\arg\max_a Q(s,a)$ |

Q-learning's target $R_{t+1} + \gamma \max_a Q(S_{t+1}, a)$ corresponds to the *greedy policy*, regardless of whether the behavior policy actually took the greedy action.

This means Q-learning directly approximates $Q^*$ — the optimal action-value function — no matter how much the agent explores. The exploration policy is a tool for data collection; it does not affect what Q-learning converges to.

**Contrast with SARSA:** SARSA evaluates the behavior policy (including exploration noise). Q-learning evaluates the greedy target policy. Same data, fundamentally different targets.

---

## Convergence: Watkins & Dayan (1992)

Q-learning converges to the optimal $Q^*$ with probability 1 given:

1. **Sufficient exploration:** every state-action pair $(s,a)$ is visited infinitely often
2. **Decaying step sizes:** Robbins-Monro conditions: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$
3. **Bounded rewards:** $|R_t| \leq R_{\max} < \infty$

**Proof sketch (Watkins & Dayan 1992):**

Define the sequence of Q-estimates $\{Q_t\}$. The update operator can be written as:

$$Q_{t+1}(s,a) = (1 - \alpha_t) Q_t(s,a) + \alpha_t \bigl[R + \gamma \max_{a'} Q_t(s',a')\bigr]$$

This is a stochastic approximation of the fixed-point equation $Q = \mathcal{T}^* Q$, where $\mathcal{T}^*$ is the Bellman optimality operator. Since $\mathcal{T}^*$ is a $\gamma$-contraction in the $\ell^\infty$ norm and the Robbins-Monro conditions ensure the noise terms average out, the sequence converges to the unique fixed point $Q^*$.

Under constant step size $\alpha$, Q-learning does not converge to a fixed point but stays within a neighborhood of $Q^*$ — often acceptable in practice.

---

## Maximization Bias

### The Problem

Q-learning always takes $\max_a Q(S', a)$ in its target. When $Q$ estimates are noisy (as they always are during learning), the maximum over estimates is *biased upward*. This is **maximization bias**.

**Example:** Suppose the true $Q^*(s, a) = 0$ for all actions in state $s$, but the estimates $Q(s, a)$ are random due to sampling noise. The maximum of noisy zero-mean estimates is positive — introducing systematic upward bias.

Maximization bias causes Q-learning to be *overoptimistic* about action values during early training. This can slow convergence and cause erratic behavior near the optimum.

### Double Q-Learning

Double Q-learning (van Hasselt, 2010) addresses this by maintaining two separate value tables $Q_1$ and $Q_2$, updated alternately. The key idea: use one table to *select* the greedy action, and the other to *evaluate* it.

With probability 0.5, update $Q_1$:

$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q_2\bigl(S_{t+1}, \arg\max_a Q_1(S_{t+1}, a)\bigr) - Q_1(S_t, A_t)\bigr]$$

Otherwise, update $Q_2$ symmetrically (swap $Q_1 \leftrightarrow Q_2$).

**Why this eliminates bias:** The action is selected by $Q_1$ and evaluated by an independent estimate $Q_2$. Because the two estimates are independent, the upward bias from maximization is removed. Formally, $\mathbb{E}[Q_2(s', \arg\max_a Q_1(s',a))] \leq \max_a Q(s',a)$ — the estimator is now unbiased or negatively biased.

---

## Diagram: Q-Learning vs SARSA Target

```
SARSA target (on-policy):
                     ┌─── behavior policy action ───┐
Q(S_t,A_t) ←── R_{t+1} + γ · Q(S_{t+1}, A_{t+1})
                                         ↑
                            actual action taken by ε-greedy

Q-learning target (off-policy):
                     ┌─── greedy action (not necessarily taken) ───┐
Q(S_t,A_t) ←── R_{t+1} + γ · max_a Q(S_{t+1}, a)
                                    ↑
                        best action regardless of what was done
```

Same transition $(S_t, A_t, R_{t+1}, S_{t+1})$ is used by both. The difference is only in the bootstrap target.

---

## SARSA vs Q-Learning: Cliff Walking Revisited

| Property | SARSA | Q-learning |
|----------|-------|------------|
| **Policy type** | On-policy | Off-policy |
| **Target policy** | $\varepsilon$-greedy | Greedy ($\pi^*$) |
| **Cliff walking path** | Upper safe path | Optimal cliff-edge path |
| **Training reward** | Higher (avoids cliff falls) | Lower (cliff falls during exploration) |
| **Deployment reward** | Slightly suboptimal | Optimal |
| **When preferred** | Dangerous exploration | Cheap exploration |

**The fundamental tension:** During training with $\varepsilon > 0$, Q-learning's optimal target policy and $\varepsilon$-greedy behavior policy conflict. The agent learns that the cliff edge is optimal but keeps accidentally falling off it during training. SARSA learns to avoid the cliff edge because its target accounts for the accidental falls.

---

## Code Implementation

```python
import numpy as np


def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float) -> int:
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))


def q_learning(
    env,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> np.ndarray:
    """
    Off-policy TD control via Q-learning (Watkins, 1989).

    Directly approximates Q* regardless of the behavior policy.

    Parameters
    ----------
    env          : Discrete Gymnasium environment.
    num_episodes : Training episodes.
    alpha        : Step size.
    gamma        : Discount factor.
    epsilon      : Exploration probability for epsilon-greedy behavior.

    Returns
    -------
    Q : np.ndarray, shape (n_states, n_actions)
        Approximation of the optimal action-value function Q*.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Behavior: epsilon-greedy (explores the environment)
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Target: greedy max — off-policy, learns Q* directly
            if terminated:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])   # ← key difference

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            # No need to carry forward next_action — Q-learning doesn't use it

    return Q


def double_q_learning(
    env,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Double Q-learning (van Hasselt, 2010).

    Reduces maximization bias by decoupling action selection from evaluation.

    Returns
    -------
    Q1, Q2 : Two action-value tables. Use (Q1 + Q2) / 2 for the final estimate.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Behavior: epsilon-greedy over the sum Q1 + Q2
            combined = Q1[state] + Q2[state]
            action = epsilon_greedy_array(combined, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.random() < 0.5:
                # Update Q1: select action with Q1, evaluate with Q2
                best_action = int(np.argmax(Q1[next_state]))
                next_val = 0.0 if terminated else Q2[next_state, best_action]
                Q1[state, action] += alpha * (reward + gamma * next_val - Q1[state, action])
            else:
                # Update Q2: select action with Q2, evaluate with Q1
                best_action = int(np.argmax(Q2[next_state]))
                next_val = 0.0 if terminated else Q1[next_state, best_action]
                Q2[state, action] += alpha * (reward + gamma * next_val - Q2[state, action])

            state = next_state

    return Q1, Q2


def epsilon_greedy_array(q_values: np.ndarray, epsilon: float) -> int:
    if np.random.random() < epsilon:
        return np.random.randint(len(q_values))
    return int(np.argmax(q_values))


# ── Example: Cliff Walking comparison ─────────────────────────────────────────
import gymnasium as gym

env = gym.make("CliffWalking-v0")
Q_ql = q_learning(env, num_episodes=500, alpha=0.1, gamma=1.0, epsilon=0.1)
greedy_policy = np.argmax(Q_ql, axis=1)
print("Q-learning greedy policy (0=Up,1=Right,2=Down,3=Left):")
print(greedy_policy.reshape(4, 12))
env.close()
```

---

## Common Pitfalls

**Pitfall 1 — Using on-policy action in the target.**
Writing `td_target = reward + gamma * Q[next_state, next_action]` where `next_action` is sampled from the policy converts Q-learning into SARSA. Both are valid, but they converge to different targets. Q-learning must use `np.max(Q[next_state])`.

**Pitfall 2 — Maximization bias in overestimating action values.**
Q-learning systematically overestimates $Q^*$ during learning due to the max operator on noisy estimates. This can cause overconfident early policies that miss better long-term strategies. Use Double Q-learning whenever maximization bias is a concern, especially in environments with many actions or high reward variance.

**Pitfall 3 — Insufficient exploration causing non-convergence.**
Q-learning requires every $(s,a)$ pair to be visited infinitely often. With a greedy behavior policy ($\varepsilon = 0$), most state-action pairs are never visited, and Q-learning cannot converge to $Q^*$. Always use $\varepsilon > 0$ during training. For convergence guarantees, decay $\varepsilon$ to 0 under GLIE conditions.

**Pitfall 4 — Comparing Q-learning training reward with SARSA training reward.**
On risky tasks like cliff walking, Q-learning will accumulate more penalties during training (because it does not account for exploratory accidents). This is expected behavior — not a bug. Evaluate the final policies greedily to get a fair comparison.

**Pitfall 5 — Treating the Q-learning update as on-policy.**
Because Q-learning is off-policy, you can learn from experience generated by any behavior policy — including random rollouts, human demonstrations, or replayed past experience (experience replay in DQN). This is a powerful feature. Treating it as if it requires on-policy data is a missed opportunity.

---

## Connections

- **Builds on:** SARSA and on-policy TD control (Guide 02), TD(0) prediction (Guide 01), Bellman optimality equation (Module 0)
- **Leads to:** Deep Q-Network (DQN) — Q-learning with neural network function approximation (Module 05), Double DQN, Dueling DQN, Prioritized Experience Replay, Q-learning with function approximation (Module 04)
- **Related to:** Value iteration (the DP counterpart to Q-learning), fitted Q-iteration (batch RL version), distributional RL (replaces scalar Q with a distribution)

---

## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 6.5–6.7 — Q-learning, Double Q-learning, and comparison with SARSA
- Watkins, C.J.C.H. & Dayan, P. (1992). *Q-learning.* Machine Learning 8(3-4) — the convergence proof
- van Hasselt, H. (2010). *Double Q-learning.* NeurIPS — addresses maximization bias
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature — extends Q-learning to DQN with experience replay and target networks

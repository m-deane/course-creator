---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# SARSA: On-Policy TD Control

## Module 3: Temporal Difference Learning
### Reinforcement Learning

<!-- Speaker notes: This deck covers SARSA — the first full RL control algorithm in the course. Unlike TD(0) prediction, SARSA both evaluates and improves the policy. The key concept students must leave with: SARSA is on-policy because the next action used in the update comes from the same policy being followed. The word "on-policy" will be contrasted sharply with Q-learning in the next guide. -->

---

## From Prediction to Control

**TD(0) prediction** estimates $V^\pi(s)$ — how good is it to be in state $s$?

**SARSA control** estimates $Q^\pi(s,a)$ — how good is it to take action $a$ in state $s$?

```
V(s) → useful for policy evaluation
Q(s,a) → useful for policy improvement (we can act greedily: argmax_a Q(s,a))
```

> The step from $V$ to $Q$ is small mathematically — replace the state value with an action-value. But it unlocks full policy optimization without a model.

<!-- Speaker notes: Clarify why we need Q instead of V for control. With V, we can compute the greedy policy only if we have the transition model: pi*(s) = argmax_a sum_{s'} p(s'|s,a)[r + gamma*V(s')]. Without the model, we cannot compute this argmax. With Q(s,a), the greedy policy is simply argmax_a Q(s,a) — no model needed. This is why Q functions are central to model-free control. -->

---

## The SARSA Name

The algorithm is named after the five quantities used in each update:

<div class="columns">
<div>

$$\underbrace{S_t}_{S}, \underbrace{A_t}_{A}, \underbrace{R_{t+1}}_{R}, \underbrace{S_{t+1}}_{S'}, \underbrace{A_{t+1}}_{A'}$$

</div>
<div>

| Symbol | What it is |
|--------|-----------|
| $S_t$ | Current state |
| $A_t$ | Action taken |
| $R_{t+1}$ | Reward received |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action (on-policy) |

</div>
</div>

> $A_{t+1}$ is chosen from the current behavior policy, not computed as the greedy max. This single choice defines on-policy learning.

<!-- Speaker notes: The name SARSA is a mnemonic for exactly what goes into the update. Have students say it: S-A-R-S-A. The critical element is the final A: it is a real action drawn from the current policy, with its exploration noise intact. Q-learning replaces this with max_a Q(s',a) — that is the entire algorithmic difference between on-policy and off-policy TD control. -->

---

## SARSA Update Rule

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \underbrace{Q(S_{t+1}, A_{t+1})}_{\text{on-policy next}} - Q(S_t, A_t)\bigr]$$

**TD error:**

$$\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

Compare to TD(0) for $V$:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

The only change: $V(S_{t+1}) \to Q(S_{t+1}, A_{t+1})$ — action value of the *actual* next action.

<!-- Speaker notes: Put both equations side by side and highlight the single difference. V(S_{t+1}) is replaced by Q(S_{t+1}, A_{t+1}). This small change has major implications: we now need to observe not just the next state, but also the next action. That observation is what makes the algorithm on-policy — we see what the agent actually does next. -->

---

## On-Policy: What Does It Mean?

**On-policy:** the policy used to generate behavior and the policy being evaluated/improved are the same.

```
Behavior policy (what agent does):   ε-greedy over Q
Target policy   (what Q represents): ε-greedy over Q
                                      ↑ Same policy!
```

SARSA evaluates the $\varepsilon$-greedy policy — including its random exploratory actions.

> SARSA asks: "How good am I, counting all the times I explore randomly?"

<!-- Speaker notes: Contrast this with what Q-learning does: Q-learning evaluates the greedy policy while behaving epsilon-greedily. So Q-learning asks: "How good would I be if I always picked the best action?" — even while it is actually exploring. On-policy vs off-policy is one of the most important distinctions in RL. It affects safety, convergence, and the value estimates the algorithm produces. -->

---

## Expected SARSA: Lower Variance

Instead of sampling $A_{t+1}$, take the expectation over all possible next actions:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_{a} \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

| Algorithm | TD Target | Variance |
|-----------|-----------|----------|
| SARSA | $R + \gamma Q(S', A')$ | Higher (samples $A'$) |
| Expected SARSA | $R + \gamma \mathbb{E}_\pi[Q(S', \cdot)]$ | Lower (averages $A'$) |
| Q-learning | $R + \gamma \max_a Q(S', a)$ | Lower (but off-policy) |

> When $\pi$ is greedy, Expected SARSA = Q-learning.

<!-- Speaker notes: Expected SARSA is strictly better than SARSA in terms of variance at the cost of a little more computation (need to sum over all actions). In practice, with small discrete action spaces, Expected SARSA is often preferred. With large or continuous action spaces, the sum is intractable and you fall back to sampled SARSA or use other methods. -->

---

## Cliff Walking: The Classic Comparison

```
 Start [S]                                Goal [G]
 ┌─────────────────────────────────────────────┐
 │  .  .  .  .  .  .  .  .  .  .  .  .  G  │  Row 0
 │  .  .  .  .  .  .  .  .  .  .  .  .  .  │  Row 1
 │  .  .  .  .  .  .  .  .  .  .  .  .  .  │  Row 2 ← SARSA path
 │  S  C  C  C  C  C  C  C  C  C  C  C  G  │  Row 3
 └─────────────────────────────────────────────┘
      ←──────── Cliff (−100) ────────────→
```

- Each step: reward $-1$
- Step onto cliff: reward $-100$, return to Start
- Reach Goal: episode ends

<!-- Speaker notes: Draw this on the board. The cliff is the bottom row between S and G. The optimal greedy path hugs the cliff edge for minimum steps. But during epsilon-greedy training, occasional random steps push you off the cliff. SARSA learns to avoid this by using the upper paths. Q-learning learns the optimal greedy path because it evaluates greedy behavior, even while behaving epsilon-greedily. -->

---

## Cliff Walking: SARSA vs Q-learning

<div class="columns">
<div>

**SARSA**
- Learns to take upper path (safe)
- Accounts for epsilon-greedy's random steps
- Higher average reward during training
- Converges to optimal safe policy

</div>
<div>

**Q-learning**
- Learns cliff-edge path (optimal greedy)
- Ignores exploration noise in target
- Lower average reward during training
- Converges to globally optimal policy

</div>
</div>

> Which is "better" depends on context. SARSA wins when exploration is dangerous. Q-learning wins when you want the optimal deployment policy.

<!-- Speaker notes: The cliff walking example is a perfect illustration of the on-policy vs off-policy tradeoff. In robotics, the cliff is a physical fall — SARSA's conservatism is essential. In a chess game, exploration is cheap — Q-learning's optimism is fine. Ask students: "In algorithmic trading, which would you prefer?" The answer depends on whether random exploratory trades have high transaction costs or slippage risk. -->

---

## SARSA Algorithm

```
Initialize Q(s,a) = 0 for all s ∈ S, a ∈ A
For each episode:
    Observe S_0
    Choose A_0 ← ε-greedy(Q, S_0)
    For t = 0, 1, 2, ...:
        Execute A_t
        Observe R_{t+1}, S_{t+1}
        Choose A_{t+1} ← ε-greedy(Q, S_{t+1})   ← on-policy choice
        Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
        S_t ← S_{t+1}, A_t ← A_{t+1}
        If S_{t+1} is terminal: break
```

Key: $A_{t+1}$ is chosen **before** the Q update and **carried forward** to the next iteration.

<!-- Speaker notes: Walk through this pseudocode step by step. The critical implementation detail: A_{t+1} must be sampled before the update and then reused at the start of the next iteration. This is the "carry-forward" pattern. Many student implementations break here — they re-sample a new action at the start of each iteration, which is subtly wrong (it changes the on-policy guarantee). -->

---

## Code: SARSA Core

```python
import numpy as np

def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[state]))

def sarsa(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)   # Choose A_0

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)  # On-policy

            target = reward if terminated else reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (target - Q[state, action])

            if terminated or truncated:
                break
            state, action = next_state, next_action   # Carry forward
    return Q
```

<!-- Speaker notes: Two lines are critical. Line "next_action = epsilon_greedy(...)": this is where on-policy is enforced — we sample from the behavior policy, not take the max. Line "state, action = next_state, next_action": this carries the sampled action forward so it is executed (not re-sampled) at the top of the next iteration. Ask students to identify where this would need to change to become Q-learning. -->

---

<!-- _class: lead -->

# Common Pitfalls

<!-- Speaker notes: The SARSA pitfalls are mostly about getting the on-policy mechanics right. The most important one: accidentally implementing Q-learning while thinking you implemented SARSA. This is surprisingly easy to do. -->

---

## Pitfall 1: Using Max Instead of On-Policy Action

```python
# SARSA (correct): uses the actual next action from the policy
next_action = epsilon_greedy(Q, next_state, epsilon)
target = reward + gamma * Q[next_state, next_action]

# Q-learning (wrong if you think you're writing SARSA):
target = reward + gamma * np.max(Q[next_state])   # ← off-policy!
```

Using `np.max` converts SARSA into Q-learning. Both are valid algorithms, but they have different convergence properties and different behavior on tasks like cliff walking.

<!-- Speaker notes: Show this side by side. A single character change (max vs actual action) converts one algorithm into the other. This is both beautiful (the algorithms are almost identical) and dangerous (the bugs are subtle). On cliff walking, these two lines produce dramatically different learned behaviors — a compelling way to demonstrate why this distinction matters. -->

---

## Pitfall 2: Re-Sampling the Action Each Iteration

```python
# WRONG: re-samples action at each step (breaks carry-forward)
while not done:
    action = epsilon_greedy(Q, state, epsilon)  # ← should NOT re-sample
    next_state, reward, done, _, _ = env.step(action)
    next_action = epsilon_greedy(Q, next_state, epsilon)
    Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
    state = next_state
    # next_action is computed but never carried forward!
```

$A_{t+1}$ must be selected once, used in the update, then *executed* in the next step.

<!-- Speaker notes: This is the most common student implementation bug. The fix: initialize action before the loop, carry (state, action) = (next_state, next_action) at the end of each iteration. The wrong version technically still runs but it is not SARSA — it produces a subtly different update that does not have SARSA's convergence guarantees. -->

---

## Pitfall 3: Epsilon Too High Throughout Training

| $\varepsilon$ | Effect |
|-----------|--------|
| 0.1 (fixed) | Converges to optimal $\varepsilon$-greedy policy, not $\pi^*$ |
| Decaying: $\varepsilon_t = 1/t$ | Converges to $\pi^*$ (GLIE conditions satisfied) |
| 0.0 | No exploration; gets stuck in local optima |

**GLIE:** Greedy in the Limit with Infinite Exploration
- Every state-action pair visited infinitely often: $\sum_t \mathbf{1}[S_t=s, A_t=a] = \infty$
- Policy converges to greedy: $\varepsilon_t \to 0$

<!-- Speaker notes: GLIE is the convergence condition for on-policy control algorithms. Fixed epsilon satisfies the exploration requirement but not the greedy limit — so SARSA with fixed epsilon converges to the best epsilon-greedy policy, which is not the same as the optimal policy. Decaying epsilon satisfies both conditions. In practice, many practitioners use fixed epsilon and accept convergence to the near-optimal epsilon-greedy policy — it works well enough for most tasks. -->

---

## Summary

<div class="columns">
<div>

### SARSA Facts
- On-policy TD control
- Uses $(S, A, R, S', A')$ tuple
- $A'$ sampled from behavior policy
- Converges to optimal $\varepsilon$-greedy $Q^\pi$
- Conservative on risky tasks

</div>
<div>

### Update Rule
$$Q(S,A) \leftarrow Q(S,A)$$
$$+ \alpha[R + \gamma Q(S',A') - Q(S,A)]$$

Expected SARSA reduces variance:
$$+ \alpha[R + \gamma \mathbb{E}_\pi Q(S',\cdot) - Q(S,A)]$$

</div>
</div>

**Next:** Q-learning replaces $Q(S', A')$ with $\max_a Q(S', a)$ — one change, major consequences.

<!-- Speaker notes: Close by building anticipation for Q-learning. Tell students: "SARSA and Q-learning differ by exactly one word in the update equation — yet they behave completely differently on cliff walking and have different convergence targets." That one word: 'actual next action' vs 'max over all next actions'. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- TD(0) prediction (Guide 01)
- Epsilon-greedy policy
- Action-value function $Q(s,a)$
- MDP notation (Module 0)

</div>
<div>

### Leads To
- Q-learning off-policy control (Guide 03)
- SARSA(λ) eligibility traces (Guide 04)
- Actor-Critic methods (Module 06)
- Safe RL (penalizing exploration risk)

</div>
</div>

**Cliff walking summary:** SARSA is safer during training; Q-learning learns a better deployment policy. Neither is universally superior.

<!-- Speaker notes: The SARSA vs Q-learning tension will appear throughout the course. Actor-critic methods essentially use a variant of on-policy updates (like SARSA) for stability. Off-policy methods like DQN use Q-learning updates. The tradeoff between training safety and policy optimality is a recurring theme in applied RL. -->


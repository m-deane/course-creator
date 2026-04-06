---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Monte Carlo Control

## Finding $\pi^*$ Without a Model

**Module 2 — Monte Carlo Methods**
Reinforcement Learning

<!-- Speaker notes: MC prediction answered "how good is this policy?" Now we tackle "how do we find the best policy?" This is the control problem. The key shift: we now need Q(s,a) instead of V(s), because without a model we cannot extract a greedy policy from V alone. This is a conceptually important distinction that distinguishes model-free from model-based methods. -->

---

## From Prediction to Control

**Prediction** (Guide 01): Given $\pi$, estimate $V^\pi$ or $Q^\pi$.

**Control**: Find $\pi^*$ without knowing $p(s', r \mid s, a)$.

**Why we can't use $V^\pi$ for model-free improvement:**

$$\pi'(s) = \arg\max_a \underbrace{\sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]}_{\text{requires model}}$$

**Why $Q^\pi$ enables model-free improvement:**

$$\pi'(s) = \arg\max_a Q^\pi(s, a) \quad \leftarrow \text{no model needed}$$


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: This slide is the conceptual linchpin of the entire guide. Draw out both formulas. The DP improvement formula has p(s',r|s,a) in it — that's the transition model. Without it, we cannot evaluate which action is best from V(s) alone. Q(s,a) bakes the action into the estimate directly, so the argmax over Q is purely a table lookup. This is why model-free RL almost always works with Q rather than V. -->

---

## The Action-Value Function $Q^\pi$

**Formal definition:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid S_t = s, A_t = a\right]$$

Expected return when:
1. Taking action $a$ in state $s$ (overriding $\pi$)
2. Following $\pi$ for all subsequent steps

**Relationship to $V^\pi$:**

$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$$

**MC estimate:** Collect episodes where $(s, a)$ is visited; average returns.


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: Q(s,a) is the expected return from the state-action pair. The key phrase in the definition: "then following pi." The first action is forced (a), subsequent actions come from the policy. This makes Q^pi policy-specific — it tells you how good a is under the current policy pi, not the optimal policy. The relationship V = sum_a pi(a|s) Q(s,a) shows V is a pi-weighted average of Q values. -->

---

## Generalized Policy Iteration with MC

```
         EVALUATE              IMPROVE
  π ──────────────→ Q^π ───────────────→ π'
  ↑                                       │
  └───────────────────────────────────────┘

Evaluation:  run episodes, average returns → Q(s,a)
Improvement: π(s) ← argmax_a Q(s,a)
```

This is **GPI** from Module 1 — now with MC evaluation instead of DP.

**Key difference from full DP:**
- DP: evaluate $Q^\pi$ *exactly* before improving
- MC: improve after *every episode* (even approximate $Q$)

GPI still converges — policy improvement is guaranteed even from approximate $Q$.


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: GPI is a unifying framework across all of RL. Point back to Module 1 where we saw GPI with DP. The arrows are the same — only the evaluation step changes. Monte Carlo fills in the evaluation arrow. The fact that we improve after every episode (not after full convergence) is called "optimistic" or "online" GPI. The policy improvement theorem still applies, guaranteeing each step is non-worse. -->

---

## The Exploration Problem

**If $\pi$ is deterministic and greedy:**
- Only one action is taken per state
- $Q(s, a)$ is never updated for non-greedy actions
- Policy cannot improve away from initial choice

**Consequence:** The algorithm gets stuck at the initial policy.

```
Q(s, LEFT)  = 1.0   ← initialized randomly, happens to be high
Q(s, RIGHT) = 0.0   ← never visited → never updated
π(s) = LEFT         ← greedy choice, never changes
```

We need a mechanism to **try all actions**.


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: This is the exploration-exploitation dilemma in concrete form. If the initial Q values happen to favor a suboptimal action, and we never try other actions, we never discover the better ones. This is why Q-table initialization matters and why exploration is non-optional. Two solutions: exploring starts (force initial diversity) and epsilon-soft policies (always keep some exploration probability). -->

---

## Solution 1: Exploring Starts

**Assumption:** Each episode can start from any $(s, a)$ pair, chosen randomly with non-zero probability for all pairs.

```
Episode 1: start (S=3, A=LEFT)  → generate trajectory
Episode 2: start (S=7, A=RIGHT) → generate trajectory
Episode 3: start (S=1, A=UP)    → generate trajectory
```

**Guarantee:** Every $(s, a)$ pair is visited infinitely often → $Q(s,a)$ converges for all pairs.

**Limitation:** Only feasible in simulation. Cannot teleport a real robot to arbitrary states.

<!-- Speaker notes: Exploring starts is the theoretical "clean" solution. In a simulator, you can reset the environment to any state and force the first action. This is how Monte Carlo ES (Algorithm 5.3 in Sutton & Barto) works. The algorithm is provably convergent to pi* under exploring starts. But in real environments — robot locomotion, real-time trading, live systems — you cannot choose your starting condition. Hence epsilon-soft policies. -->

---

## Solution 2: $\epsilon$-Soft Policies

**Definition:** $\pi$ is $\varepsilon$-soft if every action has non-zero probability:

$$\pi(a \mid s) \geq \frac{\varepsilon}{|\mathcal{A}(s)|} \quad \forall s, a$$

**$\varepsilon$-greedy policy** (simplest $\varepsilon$-soft policy):

$$\pi(a \mid s) = \begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|}  & a = \arg\max_{a'} Q(s, a') \\[6pt]
\dfrac{\varepsilon}{|\mathcal{A}|}  & \text{otherwise}
\end{cases}$$

With prob $1-\varepsilon$: **exploit** (greedy)
With prob $\varepsilon$: **explore** (uniform random)

<!-- Speaker notes: Epsilon-greedy is the workhorse exploration strategy in RL. The formula: the greedy action gets probability (1 - epsilon) plus its share of the epsilon mass. All other actions get epsilon/|A| each. Check: these probabilities sum to 1. The epsilon parameter is a dial: epsilon=0 means pure greedy (no exploration), epsilon=1 means uniform random (no exploitation). We need a value in between, and we want it to decrease over time. -->

---

## On-Policy MC Control Algorithm

```
Initialize: Q(s,a) = 0, N(s,a) = 0, π = uniform ε-soft

For each episode:
  Generate trajectory using π: S₀,A₀,R₁,...,S_{T-1},A_{T-1},R_T

  G = 0
  For t = T-1 down to 0:
    G ← γG + R_{t+1}
    If (S_t, A_t) is first-visit this episode:
      N(S_t, A_t) += 1
      Q(S_t, A_t) += [G - Q(S_t, A_t)] / N(S_t, A_t)

      A* ← argmax_a Q(S_t, a)
      For all a:
        π(a|S_t) ← ε/|A|           (exploration floor)
        if a == A*: π(a|S_t) += 1-ε (add exploitation mass)
```

**Key:** Policy updated immediately after each Q update (in-episode GPI).

<!-- Speaker notes: Walk through the algorithm step by step. Point out: the policy improvement step happens inside the backward loop, immediately after each Q update. This is more aggressive than updating once per episode — we incorporate each new Q estimate into the policy as soon as we have it. The in-episode improvement is fine because we've already generated the episode; we're just computing updates post-hoc. -->

---

## GLIE: The Convergence Condition

**Greedy in the Limit with Infinite Exploration (GLIE):**

**Condition 1 — Infinite Exploration:**
$$\lim_{k \to \infty} N_k(s, a) = \infty \quad \forall (s, a)$$

**Condition 2 — Greedy in the Limit:**
$$\lim_{k \to \infty} \pi_k(a \mid s) = \mathbf{1}\!\left[a = \arg\max_{a'} Q_k(s, a')\right]$$

**Standard GLIE schedule:** $\varepsilon_k = \dfrac{1}{k}$ (decays as $1/\text{episode}$)

Under GLIE: on-policy MC control converges to $Q^* $ and $\pi^*$.

<!-- Speaker notes: GLIE is the formal convergence condition. Condition 1 says we never stop exploring — every state-action pair gets infinitely many visits. Condition 2 says we do eventually become greedy. The 1/k schedule satisfies both: epsilon goes to zero (satisfying condition 2) but goes to zero slowly enough that each action still gets probability epsilon/|A| = 1/(k*|A|) per episode, which sums to infinity over episodes (satisfying condition 1). In practice 1/k is too slow; linear or exponential decay is used. -->

---

## Epsilon Schedule: GLIE vs. Practical

<div class="columns">

**GLIE: $\varepsilon_k = 1/k$**

```
k=1:    ε = 1.00 (random)
k=10:   ε = 0.10
k=100:  ε = 0.01
k=1000: ε = 0.001
```

- Provably convergent
- Extremely slow in practice
- Most episodes early on are random

**Exponential decay**

```
ε_k = ε_max · decay^k

ε_max=1.0, decay=0.9995:
k=1000: ε ≈ 0.60
k=5000: ε ≈ 0.08
k=10000:ε ≈ 0.007
```

- Faster convergence
- Technically violates GLIE
- Works well in practice

</div>

<!-- Speaker notes: This tradeoff is ubiquitous in RL. Theoretically clean schedules (like 1/k) often perform poorly in practice. Exponential decay schedules violate GLIE's condition 1 (exploration eventually stops), but in practice the learning has converged long before epsilon reaches near zero. A reasonable heuristic: exponential decay from 1.0 to 0.01 over the first 80% of training, then hold at 0.01 for the final 20%. -->

---

## Python: $\varepsilon$-Greedy Action Selection

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def epsilon_greedy_action(Q_state, epsilon, n_actions):
    """
    Select action using epsilon-greedy strategy.

    Args:
        Q_state:  1-D array of Q(s, ·) values for current state
        epsilon:  exploration probability
        n_actions: number of available actions

    Returns:
        action: integer in [0, n_actions)
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)       # explore
    else:
        return int(np.argmax(Q_state))            # exploit


def glie_epsilon(episode_num, epsilon_min=0.01):
    """GLIE schedule: epsilon = 1/k, floored at epsilon_min."""
    return max(epsilon_min, 1.0 / episode_num)


def exponential_epsilon(episode_num, epsilon_start=1.0,
                         epsilon_end=0.01, decay_rate=0.995):
    """Exponential decay schedule."""
    return epsilon_end + (epsilon_start - epsilon_end) * \
           (decay_rate ** episode_num)
```
</div>

<!-- Speaker notes: These utility functions are copy-paste ready for the companion notebook. Point out: glie_epsilon has a floor at epsilon_min — pure 1/k goes to zero too slowly in early episodes (epsilon=1.0 in episode 1 means always random). The floor lets the algorithm start exploiting sooner while still exploring. The exponential schedule is what most practitioners use; the decay_rate hyperparameter controls how quickly exploration is reduced. -->

---

## Blackjack: MC Control in Action

Blackjack (Sutton & Barto Example 5.3) is a natural testbed:

- **State:** (player sum, dealer card, usable ace) — 200 states
- **Actions:** Hit (0) or Stick (1) — 2 actions
- **Reward:** +1 (win), 0 (draw), -1 (lose)
- **No model:** card probabilities are not given to the agent

After MC control with 500k episodes:
```
Optimal policy found:
- Stick when player sum ≥ 20 (almost always)
- Stick on 18-19 when dealer shows weak card (2-6)
- Hit aggressively when dealer shows 7-A
```

This matches the analytically known optimal Blackjack strategy.

<!-- Speaker notes: Blackjack is ideal for this example because the optimal strategy is well-known. After running MC control, students can verify their learned policy matches the optimal "basic strategy." The 200-state, 2-action space is small enough that full tabular MC works. Walk through what the learned Q-table looks like: Q(20, Stick) >> Q(20, Hit), Q(12, Hit) > Q(12, Stick) when dealer shows 7. Seeing the learned policy match the known optimum is a satisfying validation. -->

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Using $V$ not $Q$ | Cannot improve without model | Collect $(S,A,R)$ tuples; estimate $Q$ |
| Deterministic policy | $Q$ stale for non-greedy actions | Add $\varepsilon > 0$ exploration |
| $\varepsilon$ too high | Policy never converges to greedy | Decay $\varepsilon$ over training |
| $\varepsilon$ too low / zero | Stuck in local optimum | Start with $\varepsilon \geq 0.1$ |
| Q initialized too high | Optimistic bias slows convergence | Initialize $Q = 0$ or small random values |
| Forgetting episode termination | Infinite loops | Ensure `done` terminates correctly |

<!-- Speaker notes: Walk through each pitfall with a concrete symptom the student would observe. "Q stale for non-greedy actions" manifests as: the policy never changes after initialization, because the action chosen by the initial random Q is always the greedy one and never gets explored. "Optimistic initialization" is actually sometimes used intentionally (as a form of exploration) but unintentional optimism can slow learning. -->

---

## Module 2 Control Flow

```
MC Prediction (Guide 01)
  → Evaluates V^π or Q^π from episodes
  → No model, requires complete episodes
        ↓
MC Control: Exploring Starts (tonight's algorithm)
  → Learns Q^π*, requires resettable environment
        ↓
MC Control: ε-Soft (on-policy, GLIE)
  → Learns Q^π_ε, no resettable requirement
        ↓
MC Control: Importance Sampling (Guide 03)
  → Learns Q^π_target from π_behavior data (off-policy)
```

<!-- Speaker notes: This roadmap shows the progression of increasing flexibility. Each step removes a constraint: exploring starts removes the need for epsilon-soft (but adds resettable requirement), epsilon-soft removes the resettable requirement (but adds suboptimality from epsilon), importance sampling removes the requirement that we learn from our own policy's data. Each step is a response to a practical constraint. -->

---

<!-- _class: lead -->

## Summary

- Model-free control requires $Q(s,a)$, not $V(s)$ — the argmax over $Q$ is model-free
- **Exploring Starts**: guarantees coverage, requires resettable environment
- **$\varepsilon$-greedy**: keeps all actions probable; trades off explore vs. exploit
- **GLIE**: condition for convergence to $\pi^*$ — infinite exploration + greedy in limit
- **GPI** still drives improvement: evaluate $Q$, improve $\pi$, repeat

**Key equations:**
$$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\bigl[G - Q(s,a)\bigr]$$
$$\pi(s) \leftarrow \varepsilon\text{-greedy w.r.t. } Q(s, \cdot)$$

<!-- Speaker notes: Close by returning to the central insight: Q enables model-free improvement. Everything else — exploring starts, epsilon-greedy, GLIE — is machinery to ensure we estimate Q well for all state-action pairs. Next guide: importance sampling, which allows us to learn about one policy while following a different (more exploratory) one. -->

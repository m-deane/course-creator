---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Monte Carlo Prediction

## Estimating $V^\pi$ from Sample Episodes

**Module 2 — Monte Carlo Methods**
Reinforcement Learning

<!-- Speaker notes: Welcome to Monte Carlo prediction — the first truly model-free method in this course. Students coming from Module 1 know DP requires the full MDP model. Today we throw that requirement away. The only requirement: the ability to run episodes. Frame this as a paradigm shift: from exact computation to statistical estimation. -->

---

## The Core Question

**DP asked:** What is $V^\pi(s)$, given we know $p(s', r \mid s, a)$?

**MC asks:** What do we actually *get* when we follow $\pi$ from $s$?

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right]$$

Run the policy. Observe the returns. Average them.

> **No model required. No Bellman equations to solve.**


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: The Bellman equation is the foundation of DP but requires knowing transition probabilities. MC bypasses this entirely by sampling from the real (or simulated) environment. This is why MC is called "model-free." Emphasize: we are estimating the same quantity V^pi(s) — just using a different computational strategy. -->

---

## The Return $G_t$

The return is the discounted sum of future rewards from time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

For an episode terminating at step $T$:

$$G_{T-1} = R_T$$
$$G_{T-2} = R_{T-1} + \gamma G_{T-1}$$
$$G_t = R_{t+1} + \gamma G_{t+1}$$

This **backward recursion** is the key to efficient implementation.


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: The backward recursion G_t = R_{t+1} + gamma * G_{t+1} is computationally important. Instead of summing from t to T for every t (quadratic cost), we compute G once from the end of the episode in linear time. This is how both first-visit and every-visit MC are implemented efficiently. Walk through the recursion with a short example: 3-step episode. -->

---

## MC Prediction: The Big Picture

```
Episode 1:  S0 → S3 → S1 → S4 (end)   → compute G for each state
Episode 2:  S0 → S2 → S2 → S4 (end)   → compute G for each state
Episode 3:  S0 → S3 → S2 → S4 (end)   → compute G for each state
   ...
Episode N:  ...

For each state s:
  V(s) = average of all G values collected from s
```

After enough episodes, $V(s) \to V^\pi(s)$ by the **Law of Large Numbers**.


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: This diagram is the whole algorithm. The sophistication is in the details — which visits count, how returns are computed, how averages are maintained. But the conceptual picture is simple: collect episodes, compute returns, average. Ask students: what would make this converge slowly? Answer: high variance in returns, or states visited rarely. -->

---

## First-Visit vs. Every-Visit MC

<div class="columns">

**Episode:** $S_0, S_2, S_1, S_2, S_3$ (terminal)

**First-Visit MC**
Only the *first* time $s$ appears counts.

| State | Visits | Counted |
|---|---|---|
| $S_2$ | steps 1, 3 | step 1 only |

- Unbiased estimator
- i.i.d. samples per episode
- Theory-clean convergence

**Every-Visit MC**
*All* times $s$ appears count.

| State | Visits | Counted |
|---|---|---|
| $S_2$ | steps 1, 3 | both |

- Biased for finite samples
- More data per episode
- Converges asymptotically

</div>


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: The key distinction: first-visit produces independent samples because each is the first from a fresh "start" at that state within the episode. Every-visit samples within one episode are correlated (later visits are continuations of earlier visits). In practice every-visit often works just as well and gives more data. Rule of thumb: use first-visit for theoretical work, every-visit for implementation. -->

---

## First-Visit Algorithm (Sutton & Barto Algorithm 5.1)

```
Initialize V(s) = 0, Returns(s) = [] for all s ∈ S

Loop for each episode:
  Generate episode: S_0, A_0, R_1, S_1, ..., S_{T-1}, A_{T-1}, R_T

  G = 0
  For t = T-1, T-2, ..., 0:
    G ← γ·G + R_{t+1}
    If S_t ∉ {S_0, S_1, ..., S_{t-1}}:    ← first-visit check
      Append G to Returns(S_t)
      V(S_t) ← average(Returns(S_t))
```

The first-visit check: "has $S_t$ appeared earlier in this episode?"

<!-- Speaker notes: Walk through the pseudocode step by step. Emphasize the backward loop — t goes from T-1 down to 0. The first-visit check can be implemented as a set of states seen so far. Point out: V(s) is updated immediately after computing G, so by the end of the episode all first-visited states are updated. This is "batch" style learning — one update cycle per episode. -->

---

## Python: Efficient Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def mc_prediction_first_visit(env, policy, gamma=0.99, n_episodes=10_000):
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)

        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = gamma * G + reward          # backward accumulation

            if state not in visited:        # first-visit check
                visited.add(state)
                returns_sum[state]   += G
                returns_count[state] += 1

    return {s: returns_sum[s] / returns_count[s]
            for s in returns_count}
```
</div>

<!-- Speaker notes: The `reversed(episode)` is the key implementation detail. By iterating backwards, we accumulate G cheaply with one multiply and one add per step. The `visited` set implements the first-visit check in O(1) per lookup. Note: `returns_sum` and `returns_count` are maintained across all episodes, not per-episode. This accumulates statistics over the full training run. -->

---

## Incremental Mean Update

Storing all returns is memory-inefficient. Use the **incremental update**:

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl[G - V(s)\bigr]$$

This is algebraically identical to the sample mean, but uses $O(1)$ memory.

For **non-stationary** environments, use fixed step size $\alpha$:

$$V(s) \leftarrow V(s) + \alpha\bigl[G - V(s)\bigr]$$

$[G - V(s)]$ is the **prediction error** — how wrong $V(s)$ was about $G$.

<!-- Speaker notes: The incremental form is important for two reasons: memory efficiency and extensibility. The fixed step-size version is the bridge to TD learning — we'll see in Module 3 that TD replaces G_t with R + gamma*V(s'). The term [G - V(s)] is the MC equivalent of the TD error. Planting this seed now helps students see the continuity across methods. -->

---

## No Bootstrapping: Why It Matters

<div class="columns">

**DP (bootstraps)**
$$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)\bigl[r + \gamma V(s')\bigr]$$

Uses current estimate of $V(s')$ — needs model, biased but low variance.

**MC (no bootstrap)**
$$V(s) \leftarrow V(s) + \frac{1}{N}[G_t - V(s)]$$

Uses actual observed return — no model, unbiased but high variance.

</div>

**TD (Module 3)** will combine both ideas.

<!-- Speaker notes: Bootstrapping means using your current estimates to update your estimates. DP does this — V(s') is your current guess, not ground truth. MC never does this — G_t is the actual realized return. The tradeoff: bootstrapping introduces bias (your estimate of V(s') may be wrong) but reduces variance (one step vs. many steps of randomness). The bias-variance tradeoff is a recurring theme in RL. -->

---

## Requires Episodic Tasks

MC prediction has one hard requirement: **episodes must terminate**.

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} \quad \text{requires finite } T$$

**Valid settings:**
- Games (chess, Go, Atari with lives)
- Navigation tasks with goal/failure states
- Financial episodes (trading sessions)

**Invalid settings:**
- Robots running continuously
- Server load balancing (no natural end)
- Continuing control tasks

For continuing tasks: use TD methods (Module 3) or artificially episodize.

<!-- Speaker notes: This is a fundamental limitation, not just an inconvenience. The return G_t literally cannot be computed until the episode ends. You cannot update V(s) mid-episode because you don't know G_t yet. This is why MC has high latency: long episodes = long wait per update. TD methods solve this by updating after every step using the one-step return estimate. -->

---

## Convergence Guarantee

Let $G^{(1)}, G^{(2)}, \ldots$ be first-visit returns from state $s$.

**Assumptions:**
1. Each $G^{(i)}$ has finite variance: $\text{Var}[G^{(i)}] = \sigma^2_s < \infty$
2. State $s$ is visited infinitely often as episodes accumulate

**Strong Law of Large Numbers:**

$$\frac{1}{N(s)} \sum_{i=1}^{N(s)} G^{(i)} \xrightarrow{a.s.} \mathbb{E}_\pi[G_t \mid S_t = s] = V^\pi(s)$$

Convergence is **almost sure** (with probability 1).

<!-- Speaker notes: The mathematical guarantee is simple: the sample mean of i.i.d. random variables converges to their expectation. The conditions are mild — finite variance holds whenever rewards are bounded, and infinite visits holds for any stochastic policy on a communicating MDP. The practical implication: if convergence is slow, either variance is high (need more episodes) or the state is visited rarely (policy needs adjustment). -->

---

## Variance: The Practical Challenge

MC returns have variance that grows with episode length:

$$\text{Var}[G_t] \approx \sum_{k=0}^{T-t-1} \gamma^{2k} \sigma_R^2$$

**Consequences:**
- Long episodes → high variance → slow convergence
- Stochastic rewards → noisy value estimates
- Early timesteps have highest variance (aggregate many steps)

**Mitigations:**
- Use $\gamma < 1$ (down-weights distant, variable rewards)
- Collect many episodes (variance decreases as $1/N$)
- Importance sampling with control variates (advanced)

<!-- Speaker notes: Variance is the central weakness of MC. Each return G_t is a sum of T-t random variables — variance adds up. Compare to TD which only accumulates variance over one step. The 1/N reduction from averaging is real but slow: to halve the standard error, you need 4x the episodes. This motivates variance reduction techniques and ultimately TD methods. -->

---

## Example: Blackjack Value Function

Blackjack is a classic MC prediction testbed (Sutton & Barto Example 5.1):

- **State:** (player sum, dealer card showing, has usable ace)
- **Policy $\pi$:** Stick on 20 or 21, hit otherwise
- **MC estimate after 10,000 episodes vs. 500,000 episodes**

```
10,000 episodes:   Noisy surface, high variance
500,000 episodes:  Smooth surface, matches true V^π
```

The value surface shows: high values for player sums near 20-21, low for sums ≤ 15 (likely to bust or lose to dealer).

<!-- Speaker notes: Blackjack is pedagogically ideal because the true value function can be computed analytically, so we can compare MC estimates against ground truth. Run this example in the companion notebook. Key observation: the value surface with 10k episodes is visibly noisy — students can see variance directly. 500k episodes produces a smooth surface matching the analytic solution. This makes the convergence statement tangible. -->

---

## Common Pitfalls Summary

| Pitfall | Root Cause | Fix |
|---|---|---|
| Slow convergence | High return variance | More episodes; smaller $\gamma$ |
| Missing state estimates | State never visited | Use exploring/stochastic policy |
| Wrong G values | Off-by-one indexing | Verify: $G = \gamma G + R_{t+1}$ |
| Inapplicable method | Continuing task | Switch to TD |
| Memory blow-up | Storing all returns | Use incremental mean update |

<!-- Speaker notes: Go through each pitfall with a concrete example. The off-by-one indexing error is the most common implementation bug — the reward at time t+1 follows action A_t in state S_t, so when iterating backwards through (state, action, reward) tuples, the reward at index i in the backward loop is R_{t+1} for state S_t. Verify the indexing convention against the environment's step() return signature. -->

---

## Module 2 Roadmap

```
Guide 01: MC Prediction (you are here)
  ↓ We can evaluate any policy
Guide 02: MC Control
  ↓ We can find the optimal policy
Guide 03: Importance Sampling
  ↓ We can learn from a different policy's data
```

**Next:** MC Control — how to improve the policy we're evaluating, without a model, to find $\pi^*$.

<!-- Speaker notes: Position MC prediction within the module arc. Prediction answers "how good is this policy?" Control answers "what is the best policy?" The distinction maps exactly onto policy evaluation vs. policy improvement from the GPI framework in Module 1. Guide 03 introduces the off-policy setting, which is conceptually important and practically powerful. -->

---

<!-- _class: lead -->

## Summary

- MC prediction estimates $V^\pi(s)$ by **averaging actual returns** from episodes
- **First-visit** MC: only the first occurrence of $s$ per episode counts (unbiased, i.i.d.)
- **Every-visit** MC: all occurrences count (biased, more data)
- **No bootstrapping, no model** — only complete episodes required
- Converges by **Law of Large Numbers**; main challenge is **high variance**
- Fundamental limitation: **requires episodic tasks**

**Key equation:** $V(s) \leftarrow V(s) + \frac{1}{N(s)}[G_t - V(s)]$

<!-- Speaker notes: Summarize the three core ideas: (1) average actual returns — no model, (2) first vs. every visit — a choice with tradeoffs, (3) episodic requirement — a hard constraint. The key equation is the incremental mean form — students should be able to write this from memory. Preview: next we use Q(s,a) instead of V(s) to make policy improvement possible without a model. -->

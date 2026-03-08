---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# TD(λ) and Eligibility Traces

## Module 3: Temporal Difference Learning
### Reinforcement Learning

<!-- Speaker notes: TD(lambda) is the unifying framework that connects TD(0) and Monte Carlo. Students have now seen both extremes — one-step TD and full-episode MC — and this deck shows they are part of a continuous family. The lambda parameter interpolates between them. Eligibility traces are the computational mechanism that makes this interpolation online and efficient. This is also a prerequisite for understanding Generalized Advantage Estimation (GAE), used in modern policy gradient methods. -->

---

## The Spectrum: TD(0) to Monte Carlo

```
λ = 0 (TD(0))                                     λ = 1 (MC)
──────────────────────────────────────────────────────────────
1-step            2-step        n-step        Full episode

R_{t+1}+γV(S_{t+1})            ...            G_t
   Low bias?  No — high bias                  No bias
   Low variance?  Yes                         No — very high variance
```

**The dilemma:** TD(0) is fast but biased. MC is unbiased but slow and high-variance.

**TD(λ):** interpolate between them with a single parameter $\lambda \in [0,1]$.

<!-- Speaker notes: Use the spectrum visualization to make the relationship concrete. Ask: "What is the ideal number of steps to look ahead?" The answer depends on the environment. In deterministic environments, many steps work well (low variance). In stochastic environments, more steps accumulate more noise (high variance). Lambda lets you tune this tradeoff. The goal is to find the sweet spot — often lambda around 0.7-0.9 works well in practice. -->

---

## n-Step Returns

Use $n$ actual reward observations before bootstrapping:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

<div class="columns">
<div>

**n=1 (TD(0)):**
$$R_{t+1} + \gamma V(S_{t+1})$$

**n=2:**
$$R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$$

</div>
<div>

**n=∞ (MC):**
$$R_{t+1} + \gamma R_{t+2} + \cdots = G_t$$

**Update rule:**
$$V(S_t) \leftarrow V(S_t) + \alpha[G_t^{(n)} - V(S_t)]$$

</div>
</div>

> Limitation: must wait $n$ steps before updating. Not fully online.

<!-- Speaker notes: n-step returns are practical and simple. The key limitation: you must wait n steps after visiting a state before you can update its value. This delays learning. For n=1 (TD(0)), the delay is one step. For n=10, you wait 10 steps. For MC, you wait until the end of the episode. TD(lambda) uses eligibility traces to achieve the equivalent of a weighted average over all n without the delay. -->

---

## The λ-Return

Exponentially weighted average over all n-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Weights:**

| n | Weight |
|---|--------|
| 1 | $(1-\lambda)\lambda^0 = 1-\lambda$ |
| 2 | $(1-\lambda)\lambda^1$ |
| 3 | $(1-\lambda)\lambda^2$ |
| $n$ | $(1-\lambda)\lambda^{n-1}$ |

Weights sum to 1: $\displaystyle\sum_{n=1}^\infty (1-\lambda)\lambda^{n-1} = 1$

<!-- Speaker notes: The weights are geometric — each additional step gets (1-lambda)*lambda^{n-1} weight. The factor (1-lambda) normalizes so they sum to 1. When lambda=0, all weight falls on n=1 (TD(0)). When lambda=1, the weights become equal geometric series that reproduce the full return (MC). For intermediate lambda, the weighting favors shorter returns but incorporates information from longer ones. -->

---

## Special Cases of TD(λ)

<div class="columns">
<div>

**λ = 0: TD(0)**

$$G_t^{\lambda=0} = G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$$

Weight 1.0 on 1-step return.
All other returns: weight 0.

</div>
<div>

**λ = 1: Monte Carlo**

$$G_t^{\lambda=1} = G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$

Equal-weighted sum of all n-step returns collapses to the full return.

</div>
</div>

> Both TD(0) and Monte Carlo are special cases of a single family, unified by λ.

<!-- Speaker notes: Show the mathematical derivation for lambda=1. When lambda=1, the lambda-return becomes sum_{n=1}^inf (1-1)*1^{n-1} * G^(n). This is 0*anything — but in the limit, the geometric series sums correctly to G_t. The key insight: the (1-lambda) normalization factor ensures the weights always sum to 1, and as lambda -> 1 the weight spreads uniformly across all n, reproducing the full return. -->

---

## Eligibility Traces: The Backward View

The λ-return is a *forward* concept — it requires future rewards. Eligibility traces give an equivalent *online, backward* implementation.

**Trace update at each step:**

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}(S_t = s)$$

**Value update (all states simultaneously):**

$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s), \quad \forall s$$

where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

> Traces are a **fading memory** of which states were visited recently.

<!-- Speaker notes: The eligibility trace is the key data structure. Think of it as a vector that tracks "eligibility for credit" for each state. When a state is visited, its trace spikes by 1. Between visits, all traces decay by gamma*lambda. When a reward arrives (via delta_t), the credit is distributed to all states proportional to their current trace. States visited recently get more credit; states visited long ago get less. -->

---

## Trace Dynamics Visualized

```
State S visited at t=3, t=7, t=11. Reward arrives at t=12.

Time:     3    4    5    6    7    8    9    10   11   12
                                                    R arrives

Trace:    1  0.9  0.81 0.73   +1  0.9  0.81 0.73  +1   γλ=0.9

After t=12, update V(S) proportionally to trace at t=12 ≈ 0.9
(because S was last visited at t=11, one step before reward)
```

The trace remembers that $S$ was visited at $t=3$, $t=7$, and $t=11$, and the reward at $t=12$ should be credited back to all three visits with decaying weights.

<!-- Speaker notes: Walk through this example step by step. At t=3, the trace for S spikes to 1. It decays by gamma*lambda = 0.9 each step. At t=7, S is visited again: trace = 0.9^4 + 1 ≈ 1.66 (accumulating trace). It decays again. At t=11, trace spikes again. At t=12, the TD error delta is computed and V(S) is updated by alpha * delta * e_12(S). States visited recently get proportionally more credit. -->

---

## Code: TD(λ) with Traces

```python
import numpy as np

def td_lambda(env, policy, lam, num_episodes, alpha=0.1, gamma=0.99):
    V = np.zeros(env.observation_space.n)

    for _ in range(num_episodes):
        state, _ = env.reset()
        e = np.zeros(env.observation_space.n)   # Traces reset each episode

        while True:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            next_val = 0.0 if terminated else V[next_state]
            delta = reward + gamma * next_val - V[state]

            e[state] += 1.0          # Spike at current state
            V += alpha * delta * e   # Update all states by trace weight
            e *= gamma * lam         # Decay all traces

            if terminated or truncated: break
            state = next_state
    return V
```

> Three-line inner loop after computing `delta`:
> 1. Spike trace at current state
> 2. Update all values by `alpha * delta * e`
> 3. Decay all traces by `gamma * lam`

<!-- Speaker notes: Walk through the three critical lines. The order matters: spike the trace first (so the current state gets full credit), then update values (so the current state gets the full update), then decay (so the next step's trace is already faded). Swapping the order changes the algorithm subtly. The vectorized `V += alpha * delta * e` updates all states simultaneously in one numpy operation — elegant and efficient. -->

---

## Choosing λ: The Bias-Variance Tradeoff

| λ | Bias | Variance | Practical behavior |
|---|------|----------|--------------------|
| 0.0 | High | Low | Fast updates, may underlearn |
| 0.3 | Medium | Medium-low | Good for stochastic environments |
| 0.7 | Low-medium | Medium | Common practical choice |
| 0.9 | Low | High | Near-MC; good for deterministic |
| 1.0 | None | Very high | MC; needs many episodes |

**Empirical finding (Sutton, 1988):** On random walk problems, intermediate λ ∈ [0.3, 0.9] consistently beats both endpoints.

> Treat λ as a hyperparameter. Tune it on held-out episodes.

<!-- Speaker notes: There is no universal best lambda. In deterministic environments with short episodes, high lambda works well (low variance, many steps reliable). In stochastic environments with long episodes, lower lambda is safer (variance accumulates with each step of randomness). The practical starting point: try lambda = 0.7 or 0.9 and tune from there. Cross-validate on held-out episodes rather than the training episodes used for learning. -->

---

## Forward View vs Backward View

<div class="columns">
<div>

**Forward View**
- Conceptual: weighted avg of n-step returns
- Requires future rewards to compute
- Not online
- Used for analysis and derivations
- Mathematical object: $G_t^\lambda$

</div>
<div>

**Backward View**
- Computational: eligibility traces
- Fully online — updates every step
- Memory: $O(|\mathcal{S}|)$ trace vector
- Used for implementation
- Same result as forward view (theoretically)

</div>
</div>

> The equivalence between forward and backward views is one of the most elegant proofs in RL theory (Sutton & Barto, Chapter 12).

<!-- Speaker notes: The forward-backward equivalence means you can think in forward view (what are we learning toward?) and implement in backward view (how do we do it online?). The proof of equivalence requires some care — it holds exactly for tabular TD(lambda) with offline updates. For online updates and function approximation, the equivalence is approximate. But the eligibility trace implementation is practically correct and more efficient than the forward view. -->

---

<!-- _class: lead -->

# Common Pitfalls

<!-- Speaker notes: TD(lambda) has more implementation subtleties than TD(0) or SARSA. The most critical ones are trace resets and update ordering. These mistakes are hard to detect because the algorithm still runs — it just converges to the wrong values. -->

---

## Pitfall 1: Not Resetting Traces Between Episodes

```python
# WRONG: traces carry over between episodes
V = np.zeros(n_states)
e = np.zeros(n_states)   # Outside the episode loop!

for episode in range(num_episodes):
    state, _ = env.reset()
    # e is NOT reset — carries credit from last episode!

# CORRECT: reset traces at each episode start
for episode in range(num_episodes):
    state, _ = env.reset()
    e = np.zeros(n_states)   # Reset inside the loop
```

Stale traces from the previous episode assign credit from the new episode's rewards to states visited in the last episode. The algorithm still converges but to incorrect values.

<!-- Speaker notes: This is the most common TD(lambda) bug. It is subtle because: (1) the code runs without errors, (2) values do converge to something, (3) the error only appears when you compare the final V to V^pi computed by another method. Always initialize e = np.zeros(n_states) inside the episode loop, not outside it. -->

---

## Pitfall 2: Wrong Update Order

```python
# WRONG: update values before updating trace
delta = reward + gamma * next_val - V[state]
V += alpha * delta * e     # ← uses stale trace (current state not spiked yet)
e[state] += 1.0            # ← too late

# CORRECT: spike trace first, then update values
delta = reward + gamma * next_val - V[state]
e[state] += 1.0            # ← current state gets full credit
V += alpha * delta * e     # ← uses updated trace
e *= gamma * lam           # ← decay after update
```

With the wrong order, the current state receives a delayed credit update (it appears in the trace only next step), effectively losing one step of credit.

<!-- Speaker notes: The order is: (1) compute delta, (2) spike e[state], (3) update V, (4) decay e. This order ensures the current state gets full credit from the current TD error. Any other order produces subtly wrong updates. Have students trace through what happens when S_t = S_{t-1} = same state (visited twice in a row) with each order. -->

---

## Pitfall 3: Lambda=1 Is Not Exactly Monte Carlo

TD($\lambda=1$) with online (within-episode) updates does NOT exactly match Monte Carlo.

**Why:** During the episode, each update changes $V$. These changed values affect subsequent traces and TD errors. MC uses fixed values throughout the episode and updates only at the end.

```
MC update for V(S_t):   V(S_t) ← V(S_t) + α[G_t - V(S_t)]
                                              ↑ fixed during episode

TD(λ=1) online:  V changes at every step, affecting future δ_t values
```

> Use offline (end-of-episode) updates if you need exact equivalence to MC. For practical use, online TD(λ=1) is a reasonable approximation.

<!-- Speaker notes: This is a theoretical subtlety that matters for convergence proofs more than practice. The forward-backward equivalence holds exactly for offline updates. For online updates, TD(lambda=1) and MC give similar but not identical results. The practical implication: if you observe that TD(lambda=1) behaves slightly differently from MC on the same problem, this is expected and not a bug. -->

---

## Summary

<div class="columns">
<div>

### Key Formulas

**n-step return:**
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

**λ-return:**
$$G_t^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}$$

</div>
<div>

### Eligibility Traces

**Trace update:**
$$e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbf{1}(S_t=s)$$

**Value update:**
$$V(s) \leftarrow V(s) + \alpha\delta_t e_t(s)$$

**TD(0):** $\lambda=0$ · **MC:** $\lambda=1$

</div>
</div>

**Next:** Module 04 extends TD methods to function approximation — handling large, continuous state spaces.

<!-- Speaker notes: Close by connecting TD(lambda) to what comes next. Everything in Modules 4-7 builds on the foundation just established: TD error + eligibility traces. DQN uses a one-step TD error with a neural network. Actor-Critic uses the TD error as an advantage estimate. GAE uses a lambda-return style weighting for advantage estimation. The TD error is the common currency of modern RL. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- TD(0) prediction (Guide 01)
- Monte Carlo prediction (Module 2)
- SARSA and Q-learning (Guides 02-03)
- n-step returns

</div>
<div>

### Leads To
- SARSA(λ), Q(λ) for control
- TD(λ) with function approximation (Module 04)
- Generalized Advantage Estimation — GAE (Module 06)
- Credit assignment in deep RL

</div>
</div>

**Modern relevance:** GAE (Schulman et al., 2016) is TD(λ) applied to advantage estimation in PPO and A3C — two of the most widely deployed RL algorithms. Understanding eligibility traces is understanding GAE.

<!-- Speaker notes: GAE is the reason practitioners need to understand TD(lambda). PPO with GAE is the dominant algorithm in many production RL systems, from robotics to game playing. When students implement PPO in Module 7, they will recognize the lambda parameter and trace structure from this guide. The elegance: a 1988 theoretical construct (TD lambda) appears directly in 2023 state-of-the-art systems. -->


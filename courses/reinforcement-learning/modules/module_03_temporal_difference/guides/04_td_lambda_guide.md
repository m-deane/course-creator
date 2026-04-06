# TD(λ) and Eligibility Traces

> **Reading time:** ~11 min | **Module:** 3 — Temporal Difference | **Prerequisites:** Module 2

## In Brief

TD(λ) is a family of algorithms parameterized by $\lambda \in [0, 1]$ that interpolates between one-step TD and Monte Carlo methods. The parameter $\lambda$ controls how many steps of actual rewards are used before bootstrapping. TD(0) ($\lambda=0$) and Monte Carlo ($\lambda=1$) are both special cases of a single unified framework.

<div class="callout-key">

<strong>Key Concept:</strong> TD(λ) is a family of algorithms parameterized by $\lambda \in [0, 1]$ that interpolates between one-step TD and Monte Carlo methods. The parameter $\lambda$ controls how many steps of actual rewards are used before bootstrapping.

</div>


## Key Insight

Every one-step TD estimate can be improved by looking further ahead. n-step returns combine $n$ actual reward observations before bootstrapping, trading lower bias for higher variance as $n$ increases. The $\lambda$-return is an exponentially weighted average over all n-step returns simultaneously — this proves to often be better than any single n.

---


<div class="callout-key">

<strong>Key Point:</strong> Every one-step TD estimate can be improved by looking further ahead.

</div>

## Formal Definition

### n-Step Returns

<div class="callout-key">

<strong>Key Point:</strong> ### n-Step Returns

The n-step return from time $t$ uses $n$ actual rewards before bootstrapping with $V(S_{t+n})$:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

| $n$ | ...

</div>


The n-step return from time $t$ uses $n$ actual rewards before bootstrapping with $V(S_{t+n})$:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

| $n$ | Target | Algorithm |
|-----|--------|-----------|
| 1 | $R_{t+1} + \gamma V(S_{t+1})$ | TD(0) |
| 2 | $R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$ | 2-step TD |
| $\infty$ | $R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$ | Monte Carlo |

The n-step update rule:

$$V(S_t) \leftarrow V(S_t) + \alpha \bigl[G_t^{(n)} - V(S_t)\bigr]$$

requires waiting $n$ steps before updating — online but not as immediate as TD(0).

---

## The $\lambda$-Return (Forward View)

Instead of picking a single $n$, the $\lambda$-return takes an exponentially weighted average over all n-step returns:

<div class="callout-info">

<strong>Info:</strong> Instead of picking a single $n$, the $\lambda$-return takes an exponentially weighted average over all n-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

The w...

</div>


$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

The weights $(1-\lambda)\lambda^{n-1}$ are geometric with ratio $\lambda$, and they sum to 1:

$$\sum_{n=1}^{\infty} (1-\lambda)\lambda^{n-1} = 1$$

**Weight distribution:**
- 1-step return $G_t^{(1)}$ gets weight $(1-\lambda)$
- 2-step return $G_t^{(2)}$ gets weight $(1-\lambda)\lambda$
- n-step return $G_t^{(n)}$ gets weight $(1-\lambda)\lambda^{n-1}$

Shorter returns are weighted more heavily when $\lambda < 1$; the weight on each further step decays geometrically.

### Special Cases

- **$\lambda = 0$:** All weight on $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$ → TD(0)
- **$\lambda = 1$:** Geometric weights sum to give $G_t^{(1)} = G_t$ (the full Monte Carlo return)

---

## Eligibility Traces (Backward View)

Computing the $\lambda$-return forward in time requires storing future rewards — not truly online. Eligibility traces provide an equivalent *backward view* that is fully online: it processes each time step immediately using a trace vector that accumulates credit for recently visited states.

### Eligibility Trace Update

For each state $s$, maintain an eligibility trace $e_t(s)$ updated at every step:

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}(S_t = s)$$

- $\gamma \lambda$: trace decays by $\gamma\lambda$ each step (recency weighting)
- $\mathbf{1}(S_t = s)$: trace spikes by 1 when state $s$ is visited

### Value Update with Traces

The TD error is computed as in TD(0):

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

Then every state is updated proportionally to its eligibility trace:

$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s), \quad \forall s \in \mathcal{S}$$

---

## Why Eligibility Traces Work

When a reward arrives at time $t$, multiple past states may deserve credit for it. Eligibility traces act as a *fading memory* of which states were visited recently:

- States visited recently (high trace) receive large credit updates
- States visited long ago (low trace, due to $\gamma\lambda$ decay) receive small updates
- States never visited recently (zero trace) receive no update

The combination $\gamma\lambda$ controls the decay rate:
- Small $\gamma\lambda$ (e.g., 0.0): only the most recent state is updated → TD(0)
- Large $\gamma\lambda$ (e.g., 0.9): credit propagates far back in time → near-MC

```
Time:  t-4   t-3   t-2   t-1    t      t+1
                                 R_{t+1} arrives

State visited at:
  S_{t}  : trace = γλ^0 = 1.0   (just visited) → large update
  S_{t-1}: trace = γλ^1 = 0.9   (1 step ago)   → medium update
  S_{t-2}: trace = γλ^2 = 0.81  (2 steps ago)  → small update
  S_{t-3}: trace = γλ^3 = 0.73  (3 steps ago)  → tiny update
```

---

## Forward View vs Backward View: Equivalence

The $\lambda$-return (forward view) and eligibility traces (backward view) are mathematically equivalent for linear function approximation and tabular TD. This equivalence is one of the most elegant results in RL theory.

| | Forward View ($\lambda$-return) | Backward View (eligibility traces) |
|-|--------------------------------|-------------------------------------|
| **Conceptual** | Weighted average of n-step returns | Fading credit assignment backward |
| **Online?** | No — requires future rewards | Yes — updates every step |
| **Memory** | Store future rewards | Store trace vector $e_t(s)$ |
| **Computation** | $O(n)$ delay per update | $O(|\mathcal{S}|)$ per step |
| **Practical** | Analysis/understanding | Implementation |

---


<div class="compare">
<div class="compare-card">
<div class="header before">Forward View</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Backward View: Equivalence</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Diagram: The Backup Spectrum

```
λ=0 (TD(0)):      S_t ──→ S_{t+1}                   [1-step]
                   │         │
                  R_{t+1}  V(S_{t+1})

λ=0.5:            S_t ──→ S_{t+1} ──→ S_{t+2}        [weighted 1+2-step]
                   │         │           │
                  R_{t+1}  R_{t+2}    V(S_{t+2})

λ=0.9:            S_t ──→ ... ──→ S_{t+n}            [mostly long-range]
                   │                   │
                 many rewards     V(S_{t+n}) (small weight)

λ=1 (MC):         S_t ──→ ... ──→ S_T               [full episode]
                   │
                  G_t (no bootstrap)
```

---

## Code Implementation: TD(λ) with Eligibility Traces


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np


def td_lambda_prediction(
    env,
    policy,
    lam: float,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    TD(λ) prediction using eligibility traces (backward view).

    Equivalent to the forward-view λ-return but fully online.

    Parameters
    ----------
    env          : Discrete Gymnasium environment.
    policy       : Callable[[state], action] — policy being evaluated.
    lam          : Lambda parameter in [0, 1].
                   lam=0 reduces to TD(0); lam=1 reduces to MC.
    num_episodes : Training episodes.
    alpha        : Step size.
    gamma        : Discount factor.

    Returns
    -------
    V : np.ndarray, shape (n_states,)
        Estimated value function V^pi.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for episode in range(num_episodes):
        state, _ = env.reset()
        # Eligibility traces reset to zero at the start of each episode
        e = np.zeros(n_states)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # TD error: same as TD(0)
            next_value = 0.0 if terminated else V[next_state]
            delta = reward + gamma * next_value - V[state]

            # Update trace for current state (accumulating traces variant)
            e[state] += 1.0    # Spike at visited state

            # Update ALL states proportional to their eligibility
            V += alpha * delta * e

            # Decay all traces
            e *= gamma * lam   # Apply γλ decay after the update

            state = next_state

    return V


# ── n-step return for comparison ──────────────────────────────────────────────
def n_step_td_prediction(
    env,
    policy,
    n: int,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    n-step TD prediction.

    Parameters
    ----------
    n : Number of steps to look ahead before bootstrapping.
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    for episode in range(num_episodes):
        states, rewards = [], [None]   # rewards[0] unused; indexing from 1
        state, _ = env.reset()
        states.append(state)
        terminated = False

        t = 0
        T = float("inf")   # Episode length (unknown until terminal reached)

        while True:
            if t < T:
                action = policy(states[t])
                next_state, reward, terminated, _, _ = env.step(action)
                rewards.append(reward)
                states.append(next_state)

                if terminated:
                    T = t + 1

            # Time of state being updated
            tau = t - n + 1
            if tau >= 0:
                # Compute n-step return G_tau^(n)
                end = min(tau + n, T)
                G = sum(gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, end + 1))

                if tau + n < T:
                    G += gamma ** n * V[states[tau + n]]

                V[states[tau]] += alpha * (G - V[states[tau]])

            if tau == T - 1:
                break
            t += 1

    return V


# ── Usage example ─────────────────────────────────────────────────────────────
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
random_policy = lambda s: env.action_space.sample()

V_td0   = td_lambda_prediction(env, random_policy, lam=0.0, num_episodes=5000)
V_td5   = td_lambda_prediction(env, random_policy, lam=0.5, num_episodes=5000)
V_tdmc  = td_lambda_prediction(env, random_policy, lam=1.0, num_episodes=5000)

print("TD(0)  V:", V_td0.round(3))
print("TD(0.5) V:", V_td5.round(3))
print("TD(1)  V:", V_tdmc.round(3))
env.close()
```

</div>

---

## Choosing $\lambda$: Bias-Variance Tradeoff

| $\lambda$ | Bias | Variance | Notes |
|-----------|------|----------|-------|
| 0 (TD(0)) | High | Low | Fastest update; most biased |
| 0.5 | Medium | Medium | Often a good practical choice |
| 0.9 | Low | High | Near-MC; needs many episodes |
| 1 (MC) | None | Very high | Unbiased; high sample complexity |

**Empirical observation (Sutton, 1988):** On many random walk problems, intermediate $\lambda \in [0.3, 0.9]$ outperforms both $\lambda=0$ and $\lambda=1$ in terms of mean-squared error per episode.

There is no universally optimal $\lambda$ — treat it as a hyperparameter to tune via cross-validation on a held-out set of episodes.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1 — Not resetting traces at episode boundaries.**
Eligibility traces must be zeroed at the start of each episode. If you carry traces across episodes, credit from the previous episode's states contaminates updates in the new episode. This introduces incorrect dependencies and prevents convergence.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1 — Not resetting traces at episode boundaries.**
Eligibility traces must be zeroed at the start of each episode.

</div>

**Pitfall 2 — Confusing accumulating and replacing traces.**
Two variants exist:
- *Accumulating traces:* $e_t(s) = \gamma\lambda e_{t-1}(s) + 1$ when $S_t = s$ (adds 1 unconditionally)
- *Replacing traces:* $e_t(s) = 1$ when $S_t = s$ (resets to 1, no accumulation)

Replacing traces avoid unbounded growth in loops. Accumulating traces are theoretically correct for the forward-backward equivalence. Use replacing traces for cyclic environments.

**Pitfall 3 — Updating V before updating traces.**
The correct order: update the trace first (spike at $S_t$), then update all values using the trace. If you update $V(S_t)$ first with just $\delta_t$ and then update the trace, you double-update $S_t$ and miss the distributed update to prior states.

**Pitfall 4 — Expecting $\lambda=1$ to match Monte Carlo exactly.**
TD($\lambda=1$) with online updates and a constant step size does not exactly reproduce Monte Carlo updates because the value estimates $V$ change within an episode. The forward-backward equivalence holds exactly only for offline updates (process all steps, then update). For online TD(λ=1), the behavior approximates MC but is not identical.

**Pitfall 5 — Using large $\lambda$ with large $\gamma$ in long episodes.**
The decay factor $\gamma\lambda$ determines how quickly traces fade. With $\gamma=0.99$ and $\lambda=0.99$, the decay is $(0.99)^2 \approx 0.98$ per step — traces remain non-zero for hundreds of steps. In long episodes with many states, the memory and computation cost can become significant.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** TD(0) prediction (Guide 01), Monte Carlo prediction (Module 2), n-step returns
- **Leads to:** SARSA(λ) and Q(λ) for control, TD(λ) with function approximation (Module 04), Generalized Advantage Estimation (GAE) in Actor-Critic methods (Module 06)
- **Related to:** Temporal credit assignment problem, exponential moving averages, the Bellman equation spectrum

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 7 (n-step), Chapter 12 (eligibility traces) — comprehensive treatment of the forward and backward views
- Sutton, R. S. (1988). *Learning to predict by the methods of temporal differences.* Machine Learning 3(1) — the original TD(λ) paper with random walk experiments
- Schulman, J. et al. (2016). *High-dimensional continuous control using generalized advantage estimation.* ICLR — GAE is the policy gradient version of TD(λ), widely used in modern deep RL


---

## Cross-References

<a class="link-card" href="./04_td_lambda_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_td_zero_prediction.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>

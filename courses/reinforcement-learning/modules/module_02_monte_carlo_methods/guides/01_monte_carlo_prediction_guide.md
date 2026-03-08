# Monte Carlo Prediction

## In Brief

Monte Carlo (MC) prediction estimates the value function $V^\pi(s)$ for a given policy $\pi$ by averaging actual returns observed from sample episodes. No model of the environment is required — only experience.

## Key Insight

Where Dynamic Programming computes value functions by solving Bellman equations exactly (requiring a model), Monte Carlo simply asks: "Run the policy many times and average what you actually get." The Law of Large Numbers guarantees this converges.

---

## Formal Definition

Given a policy $\pi$ and an episodic MDP with states $\mathcal{S}$, actions $\mathcal{A}$, and discount factor $\gamma \in [0, 1)$, the true state-value function is:

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid S_t = s\right]$$

where the return $G_t$ is the discounted cumulative reward from time $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

Because the task is episodic, the sum terminates at the episode end $T$:

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

MC prediction estimates $V^\pi(s)$ as the sample mean of all observed returns from state $s$:

$$V(s) \leftarrow \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

where $N(s)$ is the number of times state $s$ has been visited.

---

## Intuitive Explanation

Imagine you want to know the average score a basketball player earns per game when playing a specific strategy. You could try to model every possible game scenario mathematically — or you could just watch 1,000 games and compute the average. Monte Carlo takes the second approach.

Each episode is one "game." The return $G_t$ from a state is the total discounted reward from that point to the end of the episode. Average enough of these, and you have a reliable estimate of $V^\pi(s)$.

---

## First-Visit vs. Every-Visit MC

A single episode may visit the same state $s$ multiple times. The two variants differ in which visits count toward the average.

### First-Visit MC

Only the **first** occurrence of state $s$ within each episode contributes a return estimate. Subsequent visits in the same episode are ignored.

```
Episode: S0, S2, S1, S2, S3 (terminal)
For state S2:
  - First visit: time step 1 → G_1 counted
  - Second visit: time step 3 → IGNORED
```

**Properties:**
- Unbiased estimator of $V^\pi(s)$
- Each sample is i.i.d. (independent and identically distributed)
- Standard convergence proofs apply directly
- Preferred in most theoretical analyses (Sutton & Barto Ch. 5)

### Every-Visit MC

**All** occurrences of state $s$ in an episode contribute return estimates.

```
Episode: S0, S2, S1, S2, S3 (terminal)
For state S2:
  - First visit: time step 1 → G_1 counted
  - Second visit: time step 3 → G_3 also counted
```

**Properties:**
- Biased for finite samples (visits within an episode are correlated)
- Converges to $V^\pi(s)$ as $N(s) \to \infty$ (consistent estimator)
- More data-efficient in practice — uses every sample
- Often preferred in implementations for its simplicity

### When to Use Each

| Situation | Recommended |
|---|---|
| Theoretical analysis, proving convergence | First-visit |
| Tabular environments, sample efficiency matters | Every-visit |
| States rarely revisited within an episode | Either (negligible difference) |
| States frequently revisited (e.g., gridworld with loops) | First-visit (avoids bias) |

---

## Algorithm: MC Prediction (First-Visit)

**Input:** Policy $\pi$, discount $\gamma$, number of episodes $N$

**Output:** Estimated $V(s)$ for all $s \in \mathcal{S}$

```
Initialize:
  V(s) = 0 for all s
  Returns(s) = empty list for all s

For each episode i = 1 to N:
  1. Generate episode using π:
     S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T, S_T

  2. Compute returns backwards:
     G = 0
     For t = T-1, T-2, ..., 0:
       G = γ * G + R_{t+1}
       If S_t not in {S_0, S_1, ..., S_{t-1}}:  # first-visit check
         Append G to Returns(S_t)
         V(S_t) = mean(Returns(S_t))
```

The backward pass for computing $G_t$ is a key implementation detail — it avoids redundant summations by accumulating $G$ from the end of the episode.

---

## Python Implementation

```python
import numpy as np
from collections import defaultdict


def generate_episode(env, policy):
    """
    Run one episode under the given policy.

    Returns a list of (state, action, reward) tuples.
    The policy maps state -> action (deterministic) or
    state -> distribution over actions (stochastic).
    """
    episode = []
    state, _ = env.reset()
    done = False

    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        done = terminated or truncated
        state = next_state

    return episode


def mc_prediction_first_visit(env, policy, gamma=0.99, n_episodes=10_000):
    """
    First-visit Monte Carlo prediction.

    Estimates V^pi(s) for all visited states by averaging
    first-visit returns across episodes.

    Args:
        env:        Gymnasium environment (episodic)
        policy:     Callable state -> action
        gamma:      Discount factor
        n_episodes: Number of episodes to collect

    Returns:
        V: dict mapping state -> estimated value
    """
    returns_sum = defaultdict(float)   # sum of returns per state
    returns_count = defaultdict(int)   # number of first-visit samples

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)

        # Collect states visited this episode (for first-visit check)
        states_in_episode = {step[0] for step in episode}
        visited = set()

        # Backward pass: accumulate G from episode end
        G = 0.0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward

            if state not in visited:          # first-visit only
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1

    # Compute mean return per state
    V = {s: returns_sum[s] / returns_count[s] for s in returns_count}
    return V


def mc_prediction_every_visit(env, policy, gamma=0.99, n_episodes=10_000):
    """
    Every-visit Monte Carlo prediction.

    Same as first-visit but counts every occurrence of a state,
    not just the first.
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(n_episodes):
        episode = generate_episode(env, policy)
        G = 0.0

        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns_sum[state] += G     # no first-visit check
            returns_count[state] += 1

    V = {s: returns_sum[s] / returns_count[s] for s in returns_count}
    return V
```

---

## Incremental Mean Update

Storing all returns and recomputing the mean is wasteful. Use the incremental update rule to maintain $V(s)$ with $O(1)$ memory per state:

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl(G - V(s)\bigr)$$

This is algebraically equivalent to computing the sample mean but processes each return as it arrives. In non-stationary problems, replace $\frac{1}{N(s)}$ with a fixed step size $\alpha$:

$$V(s) \leftarrow V(s) + \alpha\bigl(G - V(s)\bigr)$$

---

## No Bootstrapping: Contrast with DP and TD

| Method | Update Uses | Requires Model | Requires Complete Episodes |
|---|---|---|---|
| Dynamic Programming | Bellman equation (bootstraps from $V(s')$) | Yes | No |
| Monte Carlo | Actual return $G_t$ (no bootstrap) | No | Yes |
| TD Learning (next module) | One-step return $R + \gamma V(s')$ (bootstraps) | No | No |

MC uses **no bootstrapping** — the update target $G_t$ is a complete, actual return from experience. This is the defining characteristic of Monte Carlo methods. The estimate is unbiased but high-variance because $G_t$ is a sum of many random variables.

---

## Diagram: Information Flow in MC Prediction

```
Episode trajectory:
S_0 --A_0--> S_1 --A_1--> S_2 --A_2--> ... --> S_T (terminal)
      R_1          R_2          R_3                  R_T

Backward return computation:
G_{T-1} = R_T
G_{T-2} = R_{T-1} + γ · G_{T-1}
G_{T-3} = R_{T-2} + γ · G_{T-2}
  ...
G_0    = R_1     + γ · G_1

Update rule (first-visit):
V(S_0) += (1/N(S_0)) * (G_0 - V(S_0))
V(S_1) += (1/N(S_1)) * (G_1 - V(S_1))  [if S_1 ≠ S_0]
  ...

After many episodes:
V(s) → V^π(s)  by Law of Large Numbers
```

---

## Convergence: Law of Large Numbers Guarantee

Let $G_t^{(1)}, G_t^{(2)}, \ldots$ be i.i.d. samples of the return from state $s$ (first-visit MC). Each has:

$$\mathbb{E}[G_t^{(i)}] = V^\pi(s), \quad \text{Var}[G_t^{(i)}] = \sigma^2_s < \infty$$

By the Strong Law of Large Numbers:

$$\frac{1}{N} \sum_{i=1}^{N} G_t^{(i)} \xrightarrow{a.s.} V^\pi(s) \quad \text{as } N \to \infty$$

The key condition is that every state be visited infinitely often — satisfied if the policy $\pi$ is stochastic (assigns non-zero probability to all actions) and the environment is communicating.

---

## Common Pitfalls

**High variance in return estimates**
Returns $G_t$ are sums of many random rewards. Variance compounds with episode length and discount: $\text{Var}[G_t] \approx \sum_{k} \gamma^{2k} \text{Var}[R_{t+k}]$. Mitigations: use $\gamma < 1$ to down-weight distant rewards; collect more episodes; use importance-sampling variance reduction (see Guide 03).

**Requires complete episodes**
MC cannot update until the episode ends. This is a fundamental limitation: (1) inapplicable to continuing (non-episodic) tasks, (2) slow learning in environments with very long episodes, (3) no online updating mid-episode. TD methods (Module 3) solve this by bootstrapping.

**Unvisited states get no estimate**
States not reached under policy $\pi$ are never updated. This is correct behavior (we only want $V^\pi$, not $V^*$), but be aware that tabular MC only estimates values for states the policy actually visits.

**Off-by-one in return computation**
The convention $G_t = R_{t+1} + \gamma R_{t+2} + \ldots$ means the reward following action $A_t$ in state $S_t$ is indexed $R_{t+1}$. Confusing $R_t$ vs $R_{t+1}$ indexing is a frequent implementation bug. Verify your backward pass: `G = gamma * G + reward` uses the reward from the current time step.

**Averaging with warm starts**
If $V(s)$ is initialized to non-zero values and updated incrementally, early estimates reflect both the prior and data. Use $V(s) = 0$ as the default initialization unless you have domain knowledge.

---

## Connections

- **Builds on:** Markov Decision Processes (Module 0), policy evaluation concept from Dynamic Programming (Module 1)
- **Leads to:** Monte Carlo Control (Guide 02), Temporal-Difference learning (Module 3) which combines MC's model-free sampling with DP's bootstrapping
- **Related to:** Importance sampling (Guide 03) for off-policy estimation; rollout methods in planning

## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 5.1–5.2 — primary reference for this guide
- Singh & Sutton (1996), "Reinforcement learning with replacing eligibility traces" — analysis of first-visit vs every-visit convergence rates
- Precup, Sutton & Singh (2000), "Eligibility Traces for Off-Policy Policy Evaluation" — extensions to off-policy settings

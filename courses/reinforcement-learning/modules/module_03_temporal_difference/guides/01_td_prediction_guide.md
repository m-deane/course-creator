# TD(0) Prediction: Learning Value Functions Online

## In Brief

Temporal Difference (TD) prediction estimates the state-value function $V^\pi$ by combining the sampling idea of Monte Carlo with the bootstrapping idea of Dynamic Programming. The agent updates value estimates after every single step — no need to wait for an episode to end.

## Key Insight

TD learning updates toward a *target that is itself an estimate*. Instead of waiting to observe the full return $G_t$, TD substitutes the immediate reward plus a discounted estimate of the next state's value:

$$\text{TD target} = R_{t+1} + \gamma V(S_{t+1})$$

The difference between this target and the current estimate is the **TD error** — the signal that drives all learning.

---

## Formal Definition

### TD(0) Update Rule

For a policy $\pi$ being evaluated, after each transition $(S_t, A_t, R_{t+1}, S_{t+1})$:

$$V(S_t) \leftarrow V(S_t) + \alpha \bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]$$

| Symbol | Meaning |
|--------|---------|
| $V(S_t)$ | Current estimate of the value of state $S_t$ |
| $\alpha \in (0, 1]$ | Step size (learning rate) |
| $R_{t+1}$ | Reward received after leaving $S_t$ |
| $\gamma \in [0, 1)$ | Discount factor |
| $V(S_{t+1})$ | Current estimate of the value of the next state |

### TD Error

The quantity inside the brackets is the **TD error** $\delta_t$:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

- $\delta_t > 0$: the transition was *better* than expected — increase $V(S_t)$
- $\delta_t < 0$: the transition was *worse* than expected — decrease $V(S_t)$
- $\delta_t = 0$: the estimate was already consistent with the observed transition

The update can be written compactly as:

$$V(S_t) \leftarrow V(S_t) + \alpha \, \delta_t$$

---

## Intuitive Explanation

Imagine you are driving from city A to city C, passing through city B. You want to estimate how long the full journey takes.

- **Monte Carlo approach:** Drive the entire route, record the total time, then update your estimate. You get an unbiased measurement but must wait until arrival.
- **DP approach:** Use a perfect road map to compute the exact expected travel time. Requires full knowledge of road conditions.
- **TD approach:** When you reach city B, you already know how long A→B took. You also have a (possibly imperfect) estimate of how long B→C takes. Combine them immediately, without waiting to reach C.

The TD approach lets you *learn while you travel*, updating estimates at every waypoint using the best information available at that moment.

---

## Bootstrapping: Using Estimates to Update Estimates

TD methods *bootstrap* — they update a guess using another guess. This is the key distinction from Monte Carlo methods, which use actual observed returns.

```
MC target:  G_t          = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + ... (actual)
TD target:  R_{t+1} + γ V(S_{t+1})                                 (bootstrap)
DP target:  Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γ V(s')]          (expectation)
```

Bootstrapping introduces bias (the estimate $V(S_{t+1})$ may be wrong) but reduces variance (we use only one step of actual randomness instead of many). This is the **bias-variance tradeoff** at the heart of TD learning.

---

## TD vs Monte Carlo vs Dynamic Programming

| Dimension | Monte Carlo | TD(0) | Dynamic Programming |
|-----------|------------|-------|---------------------|
| **Update target** | Full return $G_t$ (unbiased) | $R_{t+1} + \gamma V(S_{t+1})$ (biased bootstrap) | Full Bellman expectation (exact) |
| **When updated** | End of episode | After every step | Per sweep over all states |
| **Model required?** | No | No | Yes (transition model $p$) |
| **Episodic only?** | Yes | No — works on continuing tasks | No |
| **Online?** | No | Yes | No |
| **Variance** | High (many random steps) | Lower (one step) | Zero (uses expectations) |
| **Bias** | None (uses true returns) | Present (bootstrap) | None (uses exact model) |
| **Data efficiency** | Lower | Higher | N/A (uses model) |

**Key takeaway:** TD sits between MC and DP — it samples from the environment (like MC) but bootstraps (like DP). This combination enables online, incremental learning without a model.

---

## Convergence Properties

Under tabular representation with a fixed policy $\pi$:

1. $V(s)$ converges to $V^\pi(s)$ for all $s$ with probability 1 if:
   - The step sizes satisfy the Robbins-Monro conditions: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$
   - Every state is visited infinitely often (sufficient exploration)
2. With a constant step size $\alpha$, $V$ does not converge exactly but tracks a moving target — useful in non-stationary environments.
3. TD(0) converges faster than Monte Carlo in terms of mean-squared error on many problems, even though it introduces bias (Sutton, 1988).

---

## Diagram: The TD Backup

```
       S_t ──── A_t ──── S_{t+1}
        │                  │
       V(S_t)           V(S_{t+1})
        │                  │
        └──── δ_t ─────────┘
             ↑
        R_{t+1} + γ V(S_{t+1}) - V(S_t)
```

TD(0) uses a **one-step backup**: only one transition is observed before updating. Compare to n-step returns and TD(λ) (Guide 04), which look further ahead.

---

## Code Implementation

```python
import numpy as np


def td_zero_prediction(
    env,
    policy,
    num_episodes: int,
    alpha: float = 0.1,
    gamma: float = 0.99,
) -> np.ndarray:
    """
    Estimate V^pi using TD(0).

    Parameters
    ----------
    env        : Gymnasium-compatible environment with discrete state space.
                 env.observation_space.n gives the number of states.
    policy     : Callable[[state], action] — the policy being evaluated.
    num_episodes : Number of episodes to run.
    alpha      : Step size (learning rate). Should satisfy Robbins-Monro
                 conditions for guaranteed convergence.
    gamma      : Discount factor in [0, 1).

    Returns
    -------
    V : np.ndarray of shape (n_states,)
        Estimated state-value function under policy pi.
    """
    n_states = env.observation_space.n
    # Initialize all values to zero (pessimistic or optimistic init also works)
    V = np.zeros(n_states)

    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Behavior follows the policy being evaluated (on-policy prediction)
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Compute TD error: how wrong was our current estimate?
            if terminated:
                # Terminal state has value 0 by definition
                td_error = reward - V[state]
            else:
                td_error = reward + gamma * V[next_state] - V[state]

            # Gradient descent step in the direction of the TD error
            V[state] += alpha * td_error

            state = next_state

    return V


# ── Example: evaluate a random policy on FrozenLake ──────────────────────────
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=True)
random_policy = lambda s: env.action_space.sample()

V_hat = td_zero_prediction(env, random_policy, num_episodes=10_000)
print("Estimated V for first 8 states:")
print(V_hat[:8].round(4))
env.close()
```

---

## Common Pitfalls

**Pitfall 1 — Bootstrapping into a terminal state.**
When $S_{t+1}$ is terminal, its value is 0 by definition. If you look up `V[terminal_state]` naively, you may get a non-zero stale value. Always check `terminated` and use `reward - V[state]` (i.e., set next-state value to zero) on terminal transitions.

**Pitfall 2 — Using a constant large step size.**
A large $\alpha$ (e.g., 0.9) makes learning noisy and can prevent convergence to a fixed point. Start with $\alpha \in [0.01, 0.1]$. For proven convergence, use a schedule such as $\alpha_t = 1/N(S_t)$ where $N(S_t)$ is the visit count.

**Pitfall 3 — Confusing TD prediction with TD control.**
TD(0) as described here *evaluates* a fixed policy — it does not improve the policy. To learn an optimal policy, you need control algorithms: SARSA (Guide 02) or Q-learning (Guide 03).

**Pitfall 4 — Forgetting that TD targets are biased.**
If $V$ is initialized far from $V^\pi$, early TD targets are highly biased. Use optimistic initialization carefully; for reward-positive tasks, initializing $V(s) = 0$ tends to be fine.

**Pitfall 5 — Comparing TD and MC returns for the same sample.**
TD error $\delta_t$ is computed from a one-step transition. Monte Carlo return $G_t$ is computed from a full episode. They are different quantities and should not be mixed in the same update.

---

## Connections

- **Builds on:** Bellman expectation equation (Module 0), Monte Carlo prediction (Module 2), MDP framework
- **Leads to:** SARSA on-policy control (Guide 02), Q-learning off-policy control (Guide 03), TD(λ) and eligibility traces (Guide 04), function approximation with TD (Module 04)
- **Related to:** Kalman filtering (Bayesian view of TD), temporal difference models in neuroscience (dopamine as TD error signal)

---

## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 6.1–6.3 — canonical derivation and analysis of TD(0)
- Sutton, R. S. (1988). *Learning to predict by the methods of temporal differences.* Machine Learning 3(1) — the original TD paper
- Dayan, P. & Niv, Y. (2008). *Reinforcement learning: The good, the bad and the ugly.* Current Opinion in Neurobiology — covers the dopamine-as-TD-error connection
